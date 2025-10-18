/*
 * mks-cairo-framebuffer.c
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <cairo-gobject.h>
#include <gtk/gtk.h>

#include "mks-cairo-framebuffer-private.h"
#include "mks-util-private.h"

#define TILE_WIDTH  128
#define TILE_HEIGHT 128

struct _MksCairoFramebuffer
{
  GObject parent_instance;

  /* The underlying surface we'll draw to */
  cairo_surface_t *surface;

  /* A GBytes that we can use to reference slices which ultimately
   * references the cairo surface. This goes against the internal design
   * of GdkTexture (which are supposed to be immutable) so that we can
   * avoid additional copies beyond the one to the GPU.
   *
   * We somewhat abuse the GdkSnapshot diffing here by giving a new memory
   * texture for tiles that changed even though they will point to the
   * same memory. That way the renderer will upload the new contents for
   * that area instead of using the previously cached texture.
   */
  GBytes *content;

  /* The GdkMemoryTexture tiles we'll export using indices
   * in the format [row0:col0,col1,.. to rowN:colN]
   */
  GPtrArray *tiles;

  /* The format our framebuffer uses and corresponding format
   * the uploaded textures will use.
   */
  cairo_format_t format;
  GdkMemoryFormat memory_format;

  /* The stride for the framebuffer so that the memory texture
   * can skip past the rest of the framebuffer data.
   */
  guint stride;

  /* Number of bytes per-pixel */
  guint bpp;

  /* The height and width the framebuffer was created with */
  guint height;
  guint width;

  /* The real width and height when tiling is taken into account */
  guint real_height;
  guint real_width;

  /* The number of tiles horizontally and vertically */
  guint n_columns;
  guint n_rows;
};

enum {
  PROP_0,
  PROP_FORMAT,
  PROP_HEIGHT,
  PROP_WIDTH,
  N_PROPS
};

static cairo_user_data_key_t invalidate_key;

static int
mks_cairo_framebuffer_get_intrinsic_width (GdkPaintable *paintable)
{
  return mks_cairo_framebuffer_get_width (MKS_CAIRO_FRAMEBUFFER (paintable));
}

static int
mks_cairo_framebuffer_get_intrinsic_height (GdkPaintable *paintable)
{
  return mks_cairo_framebuffer_get_height (MKS_CAIRO_FRAMEBUFFER (paintable));
}

static double
mks_cairo_framebuffer_get_intrinsic_aspect_ratio (GdkPaintable *paintable)
{
  double width = gdk_paintable_get_intrinsic_width (paintable);
  double height = gdk_paintable_get_intrinsic_height (paintable);

  return width / height;
}

static void
mks_cairo_framebuffer_snapshot (GdkPaintable *paintable,
                                GdkSnapshot  *snapshot,
                                double        width,
                                double        height)
{
  MksCairoFramebuffer *self = MKS_CAIRO_FRAMEBUFFER (paintable);

  gtk_snapshot_save (snapshot);
  gtk_snapshot_scale (snapshot,
                      width / (double)self->width,
                      height / (double)self->height);

  for (guint row = 0; row < self->n_rows; row++)
    {
      guint row_pos = row * self->n_columns;

      for (guint col = 0; col < self->n_columns; col++)
        {
          guint col_pos = row_pos + col;

          if G_UNLIKELY (self->tiles->pdata[col_pos] == NULL)
            {
              guint tile_y = row * TILE_HEIGHT;
              guint tile_x = col * TILE_WIDTH;

              gsize byte_offset = (tile_y * self->stride) + (tile_x * self->bpp);
              gsize n_bytes = (TILE_HEIGHT-1) * self->stride + (TILE_WIDTH * self->bpp);

              g_autoptr(GBytes) bytes = g_bytes_new_from_bytes (self->content, byte_offset, n_bytes);

              self->tiles->pdata[col_pos] =
                gdk_memory_texture_new (TILE_WIDTH,
                                        TILE_HEIGHT,
                                        self->memory_format,
                                        bytes,
                                        self->stride);
            }

          gtk_snapshot_append_texture (snapshot,
                                       self->tiles->pdata[col_pos],
                                       &GRAPHENE_RECT_INIT (col * TILE_WIDTH,
                                                            row * TILE_HEIGHT,
                                                            TILE_WIDTH,
                                                            TILE_HEIGHT));
        }
    }

  gtk_snapshot_restore (snapshot);
}

static void
paintable_iface_init (GdkPaintableInterface *iface)
{
  iface->get_intrinsic_width = mks_cairo_framebuffer_get_intrinsic_width;
  iface->get_intrinsic_height = mks_cairo_framebuffer_get_intrinsic_height;
  iface->get_intrinsic_aspect_ratio = mks_cairo_framebuffer_get_intrinsic_aspect_ratio;
  iface->snapshot = mks_cairo_framebuffer_snapshot;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (MksCairoFramebuffer, mks_cairo_framebuffer, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (GDK_TYPE_PAINTABLE, paintable_iface_init))

static GParamSpec *properties [N_PROPS];

static inline void
realign (guint *value,
         guint  alignment)
{
  guint rem = (*value) % alignment;

  if (rem != 0)
    *value += (alignment - rem);
}

static void
texture_clear (gpointer data)
{
  GdkTexture *texture = data;

  if (texture != NULL)
    g_object_unref (texture);
}

static void
mks_cairo_framebuffer_constructed (GObject *object)
{
  MksCairoFramebuffer *self = (MksCairoFramebuffer *)object;

  G_OBJECT_CLASS (mks_cairo_framebuffer_parent_class)->constructed (object);

  switch (self->format)
    {
    case CAIRO_FORMAT_ARGB32:
    case CAIRO_FORMAT_RGB24:
#if G_BYTE_ORDER == G_LITTLE_ENDIAN
      self->memory_format = GDK_MEMORY_B8G8R8A8_PREMULTIPLIED;
#else
      self->memory_format = GDK_MEMORY_A8R8G8B8_PREMULTIPLIED;
#endif
      break;

    case CAIRO_FORMAT_A8:
    case CAIRO_FORMAT_A1:
    case CAIRO_FORMAT_RGB16_565:
    case CAIRO_FORMAT_RGB30:
#if _CAIRO_CHECK_VERSION(1, 17, 2)
    case CAIRO_FORMAT_RGB96F:
    case CAIRO_FORMAT_RGBA128F:
#endif
    case CAIRO_FORMAT_INVALID:
    default:
      g_warning ("Unsupported memory format from cairo format: 0x%x",
                 self->format);
      return;
    }

  self->real_width = self->width;
  self->real_height = self->height;

  realign (&self->real_width, TILE_WIDTH);
  realign (&self->real_height, TILE_HEIGHT);

  g_assert (self->real_width % TILE_WIDTH == 0);
  g_assert (self->real_height % TILE_HEIGHT == 0);

  self->surface = cairo_image_surface_create (self->format, self->real_width, self->real_height);

  if (self->surface == NULL)
    {
      g_warning ("Cairo surface creation failed: format=0x%x width=%u height=%u",
                 self->format, self->real_width, self->real_height);
      return;
    }

  self->stride = cairo_format_stride_for_width (self->format, self->real_width);
  self->bpp = self->stride / self->real_width;

  /* Currently only 4bbp are supported */
  g_assert (self->bpp == 4);

  self->content = g_bytes_new_with_free_func (cairo_image_surface_get_data (self->surface),
                                              self->stride * self->real_height,
                                              (GDestroyNotify) cairo_surface_destroy,
                                              cairo_surface_reference (self->surface));

  self->n_columns = self->real_width / TILE_WIDTH;
  self->n_rows = self->real_height / TILE_HEIGHT;

  self->tiles = g_ptr_array_new_full (self->n_columns * self->n_rows, texture_clear);
  g_ptr_array_set_size (self->tiles, self->n_columns * self->n_rows);
}

static void
mks_cairo_framebuffer_dispose (GObject *object)
{
  MksCairoFramebuffer *self = (MksCairoFramebuffer *)object;

  g_clear_pointer (&self->content, g_bytes_unref);
  g_clear_pointer (&self->surface, cairo_surface_destroy);
  g_clear_pointer (&self->tiles, g_ptr_array_unref);

  G_OBJECT_CLASS (mks_cairo_framebuffer_parent_class)->dispose (object);
}

static void
mks_cairo_framebuffer_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  MksCairoFramebuffer *self = MKS_CAIRO_FRAMEBUFFER (object);

  switch (prop_id)
    {
    case PROP_FORMAT:
      g_value_set_enum (value, mks_cairo_framebuffer_get_format (self));
      break;

    case PROP_HEIGHT:
      g_value_set_uint (value, mks_cairo_framebuffer_get_height (self));
      break;

    case PROP_WIDTH:
      g_value_set_uint (value, mks_cairo_framebuffer_get_width (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_cairo_framebuffer_set_property (GObject      *object,
                                    guint         prop_id,
                                    const GValue *value,
                                    GParamSpec   *pspec)
{
  MksCairoFramebuffer *self = MKS_CAIRO_FRAMEBUFFER (object);

  switch (prop_id)
    {
    case PROP_FORMAT:
      self->format = g_value_get_enum (value);
      break;

    case PROP_HEIGHT:
      self->height = g_value_get_uint (value);
      break;

    case PROP_WIDTH:
      self->width = g_value_get_uint (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_cairo_framebuffer_class_init (MksCairoFramebufferClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->constructed = mks_cairo_framebuffer_constructed;
  object_class->dispose = mks_cairo_framebuffer_dispose;
  object_class->get_property = mks_cairo_framebuffer_get_property;
  object_class->set_property = mks_cairo_framebuffer_set_property;

  properties[PROP_FORMAT] =
    g_param_spec_enum ("format", NULL, NULL,
                       CAIRO_GOBJECT_TYPE_FORMAT,
                       CAIRO_FORMAT_RGB24,
                       (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

  properties[PROP_HEIGHT] =
    g_param_spec_uint ("height", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

  properties[PROP_WIDTH] =
    g_param_spec_uint ("width", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
mks_cairo_framebuffer_init (MksCairoFramebuffer *self)
{
  self->format = CAIRO_FORMAT_RGB24;
}

MksCairoFramebuffer *
mks_cairo_framebuffer_new (cairo_format_t format,
                           guint          width,
                           guint          height)
{
  g_return_val_if_fail (width > 0, NULL);
  g_return_val_if_fail (height > 0, NULL);

  return g_object_new (MKS_TYPE_CAIRO_FRAMEBUFFER,
                       "format", format,
                       "height", height,
                       "width", width,
                       NULL);
}

static void
flush_and_invalidate_on_destroy (gpointer data)
{
  g_autoptr(MksCairoFramebuffer) self = data;

  cairo_surface_flush (self->surface);
  gdk_paintable_invalidate_contents (GDK_PAINTABLE (self));
}

cairo_t *
mks_cairo_framebuffer_update (MksCairoFramebuffer *self,
                              guint                x,
                              guint                y,
                              guint                width,
                              guint                height)
{
  cairo_t *cr;
  guint col1, col2;
  guint row1, row2;

  g_return_val_if_fail (MKS_IS_CAIRO_FRAMEBUFFER (self), NULL);
  g_return_val_if_fail (self->surface != NULL, NULL);

  g_assert (self->n_columns > 0);
  g_assert (self->n_rows > 0);

  col1 = MIN (x / TILE_WIDTH, self->n_columns-1);
  col2 = MIN ((x + width) / TILE_WIDTH, self->n_columns-1);

  row1 = MIN (y / TILE_HEIGHT, self->n_rows-1);
  row2 = MIN ((y + height) / TILE_HEIGHT, self->n_rows-1);

  for (guint row = row1; row <= row2; row++)
    {
      guint row_pos = row * self->n_columns;

      for (guint col = col1; col <= col2; col++)
        {
          guint col_pos = row_pos + col;

          g_assert (col_pos < self->tiles->len);

          g_clear_object (&self->tiles->pdata[col_pos]);
        }
    }

  cr = cairo_create (self->surface);
  cairo_translate (cr, x, y);
  cairo_rectangle (cr, 0, 0, width, height);
  cairo_clip (cr);

  cairo_set_user_data (cr,
                       &invalidate_key,
                       g_object_ref (self),
                       flush_and_invalidate_on_destroy);

  return cr;
}

void
mks_cairo_framebuffer_clear (MksCairoFramebuffer *self)
{
  cairo_t *cr;

  g_return_if_fail (MKS_IS_CAIRO_FRAMEBUFFER (self));

  cr = cairo_create (self->surface);
  cairo_set_operator (cr, CAIRO_OPERATOR_SOURCE);
  cairo_rectangle (cr, 0, 0,
                   self->n_columns * TILE_WIDTH,
                   self->n_rows * TILE_HEIGHT);
  cairo_set_source_rgba (cr, 0, 0, 0, 1);
  cairo_paint (cr);
  cairo_destroy (cr);
}

cairo_format_t
mks_cairo_framebuffer_get_format (MksCairoFramebuffer *self)
{
  g_return_val_if_fail (MKS_IS_CAIRO_FRAMEBUFFER (self), 0);

  return self->format;
}

guint
mks_cairo_framebuffer_get_height (MksCairoFramebuffer *self)
{
  g_return_val_if_fail (MKS_IS_CAIRO_FRAMEBUFFER (self), 0);

  return self->height;
}

guint
mks_cairo_framebuffer_get_width (MksCairoFramebuffer *self)
{
  g_return_val_if_fail (MKS_IS_CAIRO_FRAMEBUFFER (self), 0);

  return self->width;
}

void
mks_cairo_framebuffer_copy_to (MksCairoFramebuffer *self,
                               MksCairoFramebuffer *dest)
{
  cairo_t *cr;

  g_return_if_fail (MKS_IS_CAIRO_FRAMEBUFFER (self));
  g_return_if_fail (MKS_IS_CAIRO_FRAMEBUFFER (dest));

  cr = cairo_create (dest->surface);
  cairo_set_source_surface (cr, self->surface, 0, 0);
  cairo_rectangle (cr, 0, 0, self->width, self->height);
  cairo_set_operator (cr, CAIRO_OPERATOR_SOURCE);
  cairo_fill (cr);
  cairo_destroy (cr);
}
