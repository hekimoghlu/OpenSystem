#include "pw-media-stream.h"

#include <drm_fourcc.h>
#include <fcntl.h>
#include <gtk/gtk.h>
#include <pipewire/pipewire.h>
#include <spa/debug/format.h>
#include <spa/debug/types.h>
#include <spa/param/video/format-utils.h>
#include <spa/utils/result.h>
#include <spa/pod/dynamic.h>

#ifdef GDK_WINDOWING_X11
#include <gdk/x11/gdkx.h>
#endif
#ifdef GDK_WINDOWING_WAYLAND
#include <gdk/wayland/gdkwayland.h>
#endif

#define CURSOR_META_SIZE(width, height) \
  (sizeof (struct spa_meta_cursor) + \
   sizeof (struct spa_meta_bitmap) + width * height * 4)

typedef struct
{
  int major;
  int minor;
  int micro;
} PmsPwVersion;

typedef struct
{
  uint32_t spa_format;
  uint32_t drm_format;
  GArray *modifiers;
} PmsFormat;

typedef struct
{
  GSource base;

  PwMediaStream *media_stream;
  struct pw_loop *pipewire_loop;
} PipeWireSource;

struct _PwMediaStream
{
  GtkMediaStream parent_instance;

  PmsPwVersion server_version;
  int server_version_sync;

  GdkGLContext *gl_context;
  GdkPaintable *paintable;

  GArray *formats;

  uint32_t node_id;
  int pipewire_fd;

  struct pw_context *context;
  PipeWireSource *source;

  struct pw_core *core;
  struct spa_hook core_listener;

  struct pw_stream *stream;
  struct spa_hook stream_listener;
  struct spa_video_info format;
  struct spa_source *renegotiate_event;
  gboolean connected;

  struct {
    gboolean valid;
    double x, y;
    double width, height;
  } crop;

  struct {
    gboolean valid;
    double x, y;
    double hotspot_x, hotspot_y;
    double width, height;
    GdkPaintable *paintable;
  } cursor;
};

enum
{
  PROP_0,
  PROP_PIPEWIRE_FD,
  PROP_NODE_ID,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

static void pw_media_source_paintable_init (GdkPaintableInterface *iface);

static void pw_media_source_initable_iface_init (GInitableIface *iface);

G_DEFINE_TYPE_WITH_CODE (PwMediaStream, pw_media_stream, GTK_TYPE_MEDIA_STREAM,
                         G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE,
                                                pw_media_source_initable_iface_init)
                         G_IMPLEMENT_INTERFACE (GDK_TYPE_PAINTABLE,
                                                pw_media_source_paintable_init))

/*
 * Auxiliary methods
 */

static inline gboolean
parse_pipewire_version (PmsPwVersion *dest,
                        const char   *version)
{
  return sscanf (version, "%d.%d.%d", &dest->major, &dest->minor, &dest->micro) == 3;
}

static inline gboolean
check_pipewire_version (const PmsPwVersion *version,
                        int                 major,
                        int                 minor,
                        int                 micro)
{
  if (version->major != major)
    return version->major > major;
  if (version->minor != minor)
    return version->minor > minor;
  return version->micro > micro;
}

static const struct {
  uint32_t spa_format;
  uint32_t drm_format;
  GdkMemoryFormat gdk_format;
  uint32_t bpp;
  const char *name;
} supported_formats[] = {
  { SPA_VIDEO_FORMAT_BGRA,     DRM_FORMAT_ARGB8888,      GDK_MEMORY_B8G8R8A8,             4, "ARGB8888",      },
  { SPA_VIDEO_FORMAT_RGBA,     DRM_FORMAT_ABGR8888,      GDK_MEMORY_R8G8B8A8,             4, "ABGR8888",      },
  { SPA_VIDEO_FORMAT_ARGB,     DRM_FORMAT_BGRA8888,      GDK_MEMORY_A8R8G8B8,             4, "BGRA8888",      },
  { SPA_VIDEO_FORMAT_ABGR,     DRM_FORMAT_RGBA8888,      GDK_MEMORY_A8B8G8R8,             4, "RGBA8888",      },
  { SPA_VIDEO_FORMAT_BGRx,     DRM_FORMAT_XRGB8888,      GDK_MEMORY_B8G8R8X8,             4, "XRGB8888",      },
  { SPA_VIDEO_FORMAT_RGBx,     DRM_FORMAT_XBGR8888,      GDK_MEMORY_R8G8B8X8,             4, "XBGR8888",      },
  { SPA_VIDEO_FORMAT_xRGB,     DRM_FORMAT_BGRX8888,      GDK_MEMORY_X8R8G8B8,             4, "RGBX8888",      },
  { SPA_VIDEO_FORMAT_xBGR,     DRM_FORMAT_RGBX8888,      GDK_MEMORY_X8R8G8B8,             4, "XRGB8888",      },
  { SPA_VIDEO_FORMAT_RGBA_F16, DRM_FORMAT_ABGR16161616F, GDK_MEMORY_R16G16B16A16_FLOAT,   8, "ABGR16161616F", },
  { SPA_VIDEO_FORMAT_BGR,      DRM_FORMAT_RGB888,        GDK_MEMORY_R8G8B8,               3, "RGB888",        },
  { SPA_VIDEO_FORMAT_RGB,      DRM_FORMAT_BGR888,        GDK_MEMORY_B8G8R8,               3, "BGR888",        },
  { SPA_VIDEO_FORMAT_NV12,     DRM_FORMAT_NV12,          GDK_MEMORY_R8G8B8,               0, "NV12",          },
  { SPA_VIDEO_FORMAT_YUY2,     DRM_FORMAT_YUYV,          GDK_MEMORY_R8G8B8,               0, "YUYV",          },
};

static gboolean
spa_pixel_format_to_gdk_memory_format (uint32_t         spa_format,
                                       GdkMemoryFormat *out_format,
                                       uint32_t        *out_bpp)
{
  size_t i;

  for (i = 0; i < G_N_ELEMENTS (supported_formats); i++)
    {
      if (supported_formats[i].spa_format != spa_format ||
          supported_formats[i].bpp == 0)
        continue;

      *out_format = supported_formats[i].gdk_format;
      *out_bpp = supported_formats[i].bpp;
      return TRUE;
    }

  return FALSE;
}

static gboolean
spa_pixel_format_to_drm_format (uint32_t  spa_format,
                                uint32_t *out_format)
{
  size_t i;

  for (i = 0; i < G_N_ELEMENTS (supported_formats); i++)
    {
      if (supported_formats[i].spa_format != spa_format)
        continue;

      *out_format = supported_formats[i].drm_format;
      return TRUE;
    }

  return FALSE;
}

static gboolean
drm_format_is_supported (uint32_t  drm_format,
                         size_t   *idx)
{
  size_t i;

  for (i = 0; i < G_N_ELEMENTS (supported_formats); i++)
    {
      if (supported_formats[i].drm_format == drm_format)
        {
          *idx = i;
          return TRUE;
        }
    }

  return FALSE;
}

static GArray *
query_modifiers_for_format (GdkDmabufFormats *formats,
                            uint32_t          drm_format)
{
  g_autoptr (GArray) modifiers = NULL;
  uint32_t fmt;
  uint64_t mod;

  modifiers = g_array_new (FALSE, FALSE, sizeof (uint64_t));
  for (size_t i = 0; i < gdk_dmabuf_formats_get_n_formats (formats); i++)
    {
      gdk_dmabuf_formats_get_format (formats, i, &fmt, &mod);
      if (fmt == drm_format)
        g_array_append_val (modifiers, mod);
    }

  mod = DRM_FORMAT_MOD_INVALID;
  g_array_append_val (modifiers, mod);

  return g_steal_pointer (&modifiers);
}

static void
clear_format_func (PmsFormat *format)
{
  g_clear_pointer (&format->modifiers, g_array_unref);
}

static void
query_formats_and_modifiers (PwMediaStream *self)
{
  GdkDmabufFormats *formats;
  size_t i, j;
  gboolean *handled;
  guint32 fmt;
  guint64 mod;

  formats = gdk_display_get_dmabuf_formats (gdk_display_get_default ());

  handled = g_alloca (G_N_ELEMENTS (supported_formats) * sizeof (gboolean));
  memset (handled, 0, G_N_ELEMENTS (supported_formats) * sizeof (gboolean));

  g_assert (self->gl_context != NULL);

  g_clear_pointer (&self->formats, g_array_unref);
  self->formats = g_array_new (FALSE, FALSE, sizeof (PmsFormat));
  g_array_set_clear_func (self->formats, (GDestroyNotify) clear_format_func);

  for (i = 0; i < gdk_dmabuf_formats_get_n_formats (formats); i++)
    {
      gdk_dmabuf_formats_get_format (formats, i, &fmt, &mod);

      if (drm_format_is_supported (fmt, &j) && !handled[j])
        {
          GArray *modifiers;

          handled[j] = TRUE;

          modifiers = query_modifiers_for_format (formats, fmt);
          if (modifiers->len > 0)
            {
              PmsFormat format;

              format.spa_format = supported_formats[j].spa_format;
              format.drm_format = supported_formats[j].drm_format;
              format.modifiers = g_array_ref (modifiers);

              g_debug ("Display supports format %s (%u modifiers)",
                       supported_formats[j].name,
                       format.modifiers->len);

              g_array_append_val (self->formats, format);
            }

          g_array_unref (modifiers);
        }

      if (self->formats->len == G_N_ELEMENTS (supported_formats))
        break;
    }
}

static void
remove_modifier_from_format (PwMediaStream *self,
                             uint32_t       spa_format,
                             uint64_t       modifier)
{
  size_t i;

  for (i = 0; i < self->formats->len; i++)
    {
      const PmsFormat *format = &g_array_index (self->formats, PmsFormat, i);

      if (format->spa_format != spa_format)
        continue;

      if (check_pipewire_version (&self->server_version, 0, 3, 40))
        {
          int64_t j;

          for (j = 0; j < format->modifiers->len; j++)
            {
              if (g_array_index (format->modifiers, uint64_t, j) == modifier)
                g_array_remove_index_fast (format->modifiers, j--);
            }
        }
      else
        {
          /* Remove all modifiers */
          g_array_set_size (format->modifiers, 0);
        }
    }
}

static inline struct spa_pod *
build_format_param (PwMediaStream          *self,
                    struct spa_pod_builder *builder,
                    const PmsFormat        *format,
                    gboolean                build_modifiers)
{
  struct spa_pod_frame object_frame;
  struct spa_pod *result;

  spa_pod_builder_push_object (builder, &object_frame, SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat);
  spa_pod_builder_add (builder, SPA_FORMAT_mediaType, SPA_POD_Id (SPA_MEDIA_TYPE_video), 0);
  spa_pod_builder_add (builder, SPA_FORMAT_mediaSubtype, SPA_POD_Id (SPA_MEDIA_SUBTYPE_raw), 0);
  spa_pod_builder_add (builder, SPA_FORMAT_VIDEO_format, SPA_POD_Id (format->spa_format), 0);

  if (build_modifiers && format->modifiers->len > 0)
    {
      struct spa_pod_frame modifiers_frame;
      uint32_t i;

      spa_pod_builder_prop (builder,
                            SPA_FORMAT_VIDEO_modifier,
                            SPA_POD_PROP_FLAG_MANDATORY | SPA_POD_PROP_FLAG_DONT_FIXATE);

      spa_pod_builder_push_choice (builder, &modifiers_frame, SPA_CHOICE_Enum, 0);
      spa_pod_builder_long (builder, g_array_index (format->modifiers, uint64_t, 0));
      for (i = 0; i < format->modifiers->len; i++)
        spa_pod_builder_long (builder, g_array_index (format->modifiers, uint64_t, i));
      if (spa_pod_builder_pop (builder, &modifiers_frame) == NULL)
        g_warning ("Ran out of POD space");
    }

  spa_pod_builder_add (builder,
                       SPA_FORMAT_VIDEO_size,
                       SPA_POD_CHOICE_RANGE_Rectangle (&SPA_RECTANGLE(320, 240), // Arbitrary
                                                       &SPA_RECTANGLE(1, 1),
                                                       &SPA_RECTANGLE(8192, 4320)),
                       0);

  result = spa_pod_builder_pop (builder, &object_frame);
  if (result == NULL)
    g_warning ("Ran out of POD space");

  return result;
}

static GPtrArray *
build_stream_format_params (PwMediaStream          *self,
                            struct spa_pod_builder *builder)
{
  g_autoptr (GPtrArray) params = NULL;
  uint32_t i;

  g_assert (self->formats != NULL);

  params = g_ptr_array_sized_new (2 * self->formats->len);

  for (i = 0; i < self->formats->len; i++)
    {
      const PmsFormat *format = &g_array_index (self->formats, PmsFormat, i);

      if (check_pipewire_version (&self->server_version, 0, 3, 33))
        g_ptr_array_add (params, build_format_param (self, builder, format, TRUE));

      g_ptr_array_add (params, build_format_param (self, builder, format, FALSE));
    }

  return g_steal_pointer (&params);
}

static void
renegotiate_stream_format (void     *user_data,
                           uint64_t  expirations)
{
  g_autoptr (GPtrArray) new_params = NULL;
  struct spa_pod_dynamic_builder builder;
  PwMediaStream *self = user_data;
  uint8_t params_buffer[4096];

  spa_pod_dynamic_builder_init (&builder, params_buffer, sizeof (params_buffer), 4096);
  new_params = build_stream_format_params (self, (struct spa_pod_builder *) &builder);

  pw_stream_update_params (self->stream,
                           (const struct spa_pod **)new_params->pdata,
                           new_params->len);

  spa_pod_dynamic_builder_clean (&builder);
}

static void
connect_stream (PwMediaStream *self)
{
  g_autoptr (GPtrArray) params = NULL;
  struct spa_pod_dynamic_builder builder;
  uint8_t params_buffer[4096];
  int result;

  g_assert (self->gl_context != NULL);

  if (self->connected)
    return;

  spa_pod_dynamic_builder_init (&builder, params_buffer, sizeof (params_buffer), 4096);
  params = build_stream_format_params (self, (struct spa_pod_builder *) &builder);

  result = pw_stream_connect (self->stream,
                              PW_DIRECTION_INPUT,
                              self->node_id,
                              PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS,
                              (const struct spa_pod **)params->pdata,
                              params->len);

  spa_pod_dynamic_builder_clean (&builder);

  if (result != 0)
    g_warning ("Could not connect: %s", spa_strerror (result));

  self->connected = TRUE;
}


static void
select_node (PwMediaStream *self)
{
  self->node_id = 44;
  connect_stream (self);
}

typedef struct
{
  PwMediaStream *self;
  struct pw_buffer *b;
} BufferData;

static void
dmabuf_texture_destroy (gpointer data)
{
  BufferData *bd = data;

  if (bd->self->stream)
    pw_stream_queue_buffer (bd->self->stream, bd->b);
  g_free (bd);
}

/*
 * pw_stream
 */

static void
on_process_cb (void *user_data)
{
  PwMediaStream *self = user_data;
  struct spa_meta_cursor *cursor;
  struct spa_meta_header *header;
  struct spa_meta_region *region;
  struct spa_buffer *buffer;
  struct pw_buffer *b;
  gboolean size_changed;
  gboolean invalidated;
  gboolean has_buffer;
  gboolean has_crop;
  uint32_t drm_format;

  /* Find the most recent buffer */
  b = NULL;
  while (TRUE)
    {
      struct pw_buffer *aux = pw_stream_dequeue_buffer (self->stream);

      if (!aux)
        break;
      if (b)
        pw_stream_queue_buffer (self->stream, b);
      b = aux;
  }

  if (!b)
    {
      g_debug ("Out of buffers!");
      return;
    }

  buffer = b->buffer;
  header = spa_buffer_find_meta_data (buffer, SPA_META_Header, sizeof (*header));

  if (header && (header->flags & SPA_META_HEADER_FLAG_CORRUPTED) > 0)
    {
      g_warning ("Buffer is corrupt");
      pw_stream_queue_buffer (self->stream, b);
      return;
    }

  has_buffer = buffer->datas[0].chunk->size != 0;
  size_changed = FALSE;
  invalidated = FALSE;

  if (!has_buffer)
    goto read_metadata;

  if (buffer->datas[0].type == SPA_DATA_DmaBuf)
    {
      g_autoptr(GdkDmabufTextureBuilder) builder = NULL;
      g_autoptr(GError) error = NULL;

      g_debug ("DMA-BUF info: fd:%ld, stride:%d, offset:%u, size:%dx%d, modifier:%#lx",
               buffer->datas[0].fd, buffer->datas[0].chunk->stride,
               buffer->datas[0].chunk->offset,
               self->format.info.raw.size.width,
               self->format.info.raw.size.height,
               self->format.info.raw.modifier);

      if (!spa_pixel_format_to_drm_format (self->format.info.raw.format, &drm_format))
        {
          g_critical ("Unsupported DMA buffer format: %d", self->format.info.raw.format);
          goto read_metadata;
        }

      g_debug ("DMA buffer format: %.4s", (char *) &drm_format);

      builder = gdk_dmabuf_texture_builder_new ();
      gdk_dmabuf_texture_builder_set_display (builder, gdk_display_get_default ());
      gdk_dmabuf_texture_builder_set_width (builder, self->format.info.raw.size.width);
      gdk_dmabuf_texture_builder_set_height (builder, self->format.info.raw.size.height);
      gdk_dmabuf_texture_builder_set_fourcc (builder, drm_format);
      gdk_dmabuf_texture_builder_set_modifier (builder, self->format.info.raw.modifier);
      gdk_dmabuf_texture_builder_set_n_planes (builder, buffer->n_datas);

      for (gsize i = 0; i < buffer->n_datas; i++)
        {
          gdk_dmabuf_texture_builder_set_fd (builder, i, buffer->datas[i].fd);
          gdk_dmabuf_texture_builder_set_offset (builder, i, buffer->datas[i].chunk->offset);
          gdk_dmabuf_texture_builder_set_stride (builder, i, buffer->datas[i].chunk->stride);
        }

      BufferData *bd = g_new0 (BufferData, 1);
      bd->self = self;
      bd->b = b;

      g_clear_object (&self->paintable);
      self->paintable = GDK_PAINTABLE (gdk_dmabuf_texture_builder_build (builder, dmabuf_texture_destroy, bd, &error));

      if (!self->paintable)
        {
          g_warning ("%s", error->message);

          remove_modifier_from_format (self,
                                       self->format.info.raw.format,
                                       self->format.info.raw.modifier);
          pw_loop_signal_event (self->source->pipewire_loop, self->renegotiate_event);
        }

      invalidated = TRUE;
    }
  else
    {
      g_autoptr (GdkTexture) texture = NULL;
      g_autoptr (GBytes) bytes = NULL;
      GdkMemoryFormat gdk_format;
      uint32_t bpp;
      size_t size;

      g_debug ("Buffer has memory texture");

      if (!spa_pixel_format_to_gdk_memory_format (self->format.info.raw.format,
                                                  &gdk_format,
                                                  &bpp))
        {
          g_critical ("Unsupported memory buffer format: %d", self->format.info.raw.format);
          goto read_metadata;
        }

      size = self->format.info.raw.size.width * self->format.info.raw.size.height * bpp;
      bytes = g_bytes_new (buffer->datas[0].data, size);

      texture = gdk_memory_texture_new (self->format.info.raw.size.width,
                                        self->format.info.raw.size.height,
                                        gdk_format,
                                        bytes,
                                        buffer->datas[0].chunk->stride);
      g_set_object (&self->paintable, GDK_PAINTABLE (texture));

      pw_stream_queue_buffer (self->stream, b);

      invalidated = TRUE;
    }

  /* Video Crop */
  region = spa_buffer_find_meta_data (buffer, SPA_META_VideoCrop, sizeof (*region));
  has_crop = region && spa_meta_region_is_valid (region);
  invalidated |= has_crop != self->crop.valid;
  if (has_crop)
    {
      g_debug ("Crop Region available (%dx%d+%d+%d)",
               region->region.position.x, region->region.position.y,
               region->region.size.width, region->region.size.height);

      size_changed = self->crop.width != region->region.size.width ||
                     self->crop.height != region->region.size.height;

      invalidated |= self->crop.x != region->region.position.x ||
                     self->crop.y != region->region.position.y ||
                     size_changed;

      self->crop.x = region->region.position.x;
      self->crop.y = region->region.position.y;
      self->crop.width = region->region.size.width;
      self->crop.height = region->region.size.height;
      self->crop.valid = TRUE;
    }
  else
    {
      self->crop.valid = FALSE;
    }

read_metadata:

  /* Cursor */
  cursor = spa_buffer_find_meta_data (buffer, SPA_META_Cursor, sizeof (*cursor));
  self->cursor.valid = cursor && spa_meta_cursor_is_valid (cursor);
  if (self->cursor.valid)
    {
      struct spa_meta_bitmap *bitmap = NULL;
      GdkMemoryFormat gdk_format;
      uint32_t bpp;

      if (cursor->bitmap_offset)
        bitmap = SPA_MEMBER (cursor, cursor->bitmap_offset, struct spa_meta_bitmap);

      if (bitmap)
        g_clear_object (&self->cursor.paintable);

      if (bitmap &&
          bitmap->size.width > 0 &&
          bitmap->size.height > 0 &&
          spa_pixel_format_to_gdk_memory_format (bitmap->format,
                                                 &gdk_format,
                                                 &bpp))
        {
          g_autoptr (GdkTexture) texture = NULL;
          g_autoptr (GBytes) bytes = NULL;
          const uint8_t *bitmap_data;

          bitmap_data = SPA_MEMBER (bitmap, bitmap->offset, uint8_t);
          self->cursor.width = bitmap->size.width;
          self->cursor.height = bitmap->size.height;
          self->cursor.hotspot_x = cursor->hotspot.x;
          self->cursor.hotspot_y = cursor->hotspot.y;

          g_assert (self->cursor.paintable == NULL);

          bytes = g_bytes_new (bitmap_data,
                               bitmap->size.width * bitmap->size.height * bpp);

          texture = gdk_memory_texture_new (bitmap->size.width,
                                            bitmap->size.height,
                                            gdk_format,
                                            bytes,
                                            bitmap->stride);
          g_set_object (&self->cursor.paintable, GDK_PAINTABLE (texture));

          invalidated = TRUE;
        }

      invalidated |= self->cursor.x != cursor->position.x ||
                     self->cursor.y != cursor->position.y;

      self->cursor.x = cursor->position.x;
      self->cursor.y = cursor->position.y;

      g_debug ("Stream has cursor %.0lfx%.0lf +%.0lf+%.0lf (hotspot: %.0lfx%.0lf)",
               self->cursor.x,
               self->cursor.y,
               self->cursor.width,
               self->cursor.height,
               self->cursor.hotspot_x,
               self->cursor.hotspot_y);
    }

  gtk_media_stream_update (GTK_MEDIA_STREAM (self), header ? header->pts : 0);

  if (size_changed)
    gdk_paintable_invalidate_size (GDK_PAINTABLE (self));
  if (invalidated)
    gdk_paintable_invalidate_contents (GDK_PAINTABLE (self));
}

static void
on_param_changed_cb (void                 *user_data,
                     uint32_t              id,
                     const struct spa_pod *param)
{
  struct spa_pod_builder pod_builder;
  PwMediaStream *self = user_data;
  const struct spa_pod *params[3];
  uint8_t params_buffer[1024];
  int buffer_types;
  int result;

  if (!param || id != SPA_PARAM_Format)
    return;

  result = spa_format_parse (param,
                             &self->format.media_type,
                             &self->format.media_subtype);
  if (result < 0)
    return;

  if (self->format.media_type != SPA_MEDIA_TYPE_video ||
      self->format.media_subtype != SPA_MEDIA_SUBTYPE_raw)
    return;

  spa_format_video_raw_parse (param, &self->format.info.raw);

  g_debug ("Negotiated format:");
  g_debug ("     Format: %d (%s)",
           self->format.info.raw.format,
           spa_debug_type_find_name (spa_type_video_format,
                                     self->format.info.raw.format));
    g_debug ("     Size: %dx%d",
             self->format.info.raw.size.width,
             self->format.info.raw.size.height);
    g_debug ("     Framerate: %d/%d",
             self->format.info.raw.framerate.num,
             self->format.info.raw.framerate.denom);

  /* Video crop */
  pod_builder = SPA_POD_BUILDER_INIT (params_buffer, sizeof (params_buffer));
  params[0] = spa_pod_builder_add_object (
    &pod_builder,
    SPA_TYPE_OBJECT_ParamMeta,
    SPA_PARAM_Meta,
    SPA_PARAM_META_type, SPA_POD_Id (SPA_META_VideoCrop),
    SPA_PARAM_META_size, SPA_POD_Int (sizeof (struct spa_meta_region)));

  /* Cursor */
  params[1] = spa_pod_builder_add_object (
    &pod_builder,
    SPA_TYPE_OBJECT_ParamMeta,
    SPA_PARAM_Meta,
    SPA_PARAM_META_type, SPA_POD_Id (SPA_META_Cursor),
    SPA_PARAM_META_size, SPA_POD_CHOICE_RANGE_Int (CURSOR_META_SIZE (64, 64),
                                                   CURSOR_META_SIZE (1, 1),
                                                   CURSOR_META_SIZE (1024, 1024)));

  /* Buffer options */
  buffer_types = 1 << SPA_DATA_MemPtr;

  if (check_pipewire_version (&self->server_version, 0, 3, 24) ||
      spa_pod_find_prop (param, NULL, SPA_FORMAT_VIDEO_modifier) != NULL)
    {
      buffer_types |= 1 << SPA_DATA_DmaBuf;
    }

  params[2] = spa_pod_builder_add_object (
    &pod_builder,
    SPA_TYPE_OBJECT_ParamBuffers,
    SPA_PARAM_Buffers,
    SPA_PARAM_BUFFERS_dataType, SPA_POD_Int (buffer_types));

  pw_stream_update_params (self->stream, params, 3);
}

static void
on_state_changed_cb (void                 *user_data,
                     enum pw_stream_state  old,
                     enum pw_stream_state  state,
                     const char           *error)
{
  PwMediaStream *self = user_data;

  g_debug ("Stream %p state: %s → %s (error: %s)",
           self->stream,
           pw_stream_state_as_string (old),
           pw_stream_state_as_string (state),
           error ? error : "none");

  if (old == PW_STREAM_STATE_CONNECTING && state == PW_STREAM_STATE_PAUSED)
    gtk_media_stream_stream_prepared (GTK_MEDIA_STREAM (self), FALSE, TRUE, FALSE, 0);
}

static const struct pw_stream_events stream_events = {
  PW_VERSION_STREAM_EVENTS,
  .state_changed = on_state_changed_cb,
  .param_changed = on_param_changed_cb,
  .process = on_process_cb,
};

static void
on_core_done_cb (void     *user_data,
                 uint32_t  id,
                 int       seq)
{
  PwMediaStream *self = user_data;

  if (id == PW_ID_CORE && self->server_version_sync == seq)
    self->server_version_sync = -1;
}

static void
on_core_error_cb (void       *user_data,
                  uint32_t    id,
                  int         seq,
                  int         res,
                  const char *message)
{
  PwMediaStream *self = user_data;

  gtk_media_stream_error (GTK_MEDIA_STREAM (self),
                          G_IO_ERROR,
                          G_IO_ERROR_FAILED,
                          "PipeWire stream error: %s (%d): %s",
                          g_strerror (res),
                          res,
                          message);
}

static void
on_core_info_cb (void                      *user_data,
                 const struct pw_core_info *info)
{
  PwMediaStream *self = user_data;

  g_message ("PipeWire server version: %s", info->version);

  if (!parse_pipewire_version (&self->server_version, info->version))
    g_warning ("Failed to parse PipeWire version");
}

static const struct pw_core_events core_events = {
  PW_VERSION_CORE_EVENTS,
  .done = on_core_done_cb,
  .error = on_core_error_cb,
  .info = on_core_info_cb,
};


/*
 * PipeWireSource
 */

static gboolean
pipewire_loop_source_prepare (GSource *base,
                              int     *timeout)
{
  *timeout = -1;
  return FALSE;
}

static gboolean
pipewire_loop_source_dispatch (GSource     *source,
                               GSourceFunc  callback,
                               gpointer     user_data)
{
  PipeWireSource *pw_source = (PipeWireSource *) source;
  int result;

  result = pw_loop_iterate (pw_source->pipewire_loop, 0);
  if (result < 0)
    g_warning ("pipewire_loop_iterate failed: %s", spa_strerror (result));

  return TRUE;
}

static void
pipewire_loop_source_finalize (GSource *source)
{
  PipeWireSource *pw_source = (PipeWireSource *) source;

  pw_loop_leave (pw_source->pipewire_loop);
  pw_loop_destroy (pw_source->pipewire_loop);
}

static GSourceFuncs pw_source_funcs =
{
  pipewire_loop_source_prepare,
  NULL,
  pipewire_loop_source_dispatch,
  pipewire_loop_source_finalize,
  NULL,
  NULL,
};

static PipeWireSource *
create_pipewire_source (PwMediaStream *self)
{
  PipeWireSource *pw_source;

  pw_source = (PipeWireSource *) g_source_new (&pw_source_funcs,
                                               sizeof (PipeWireSource));
  pw_source->media_stream = self;
  pw_source->pipewire_loop = pw_loop_new (NULL);
  if (!pw_source->pipewire_loop)
    {
      g_source_unref ((GSource *) pw_source);
      return NULL;
    }

  g_source_add_unix_fd (&pw_source->base,
                        pw_loop_get_fd (pw_source->pipewire_loop),
                        G_IO_IN | G_IO_ERR);

  pw_loop_enter (pw_source->pipewire_loop);
  g_source_attach (&pw_source->base, NULL);

  return pw_source;
}


/*
 * GdkPaintable interface
 */

static inline gboolean
has_effective_crop (PwMediaStream *self)
{
  return self->crop.valid &&
         (self->crop.x != 0 ||
          self->crop.y != 0 ||
          self->crop.width < self->format.info.raw.size.width ||
          self->crop.height < self->format.info.raw.size.height);
}

static void
pw_media_stream_paintable_snapshot (GdkPaintable *paintable,
                                     GdkSnapshot  *snapshot,
                                     double        width,
                                     double        height)
{
  PwMediaStream *self = PW_MEDIA_STREAM (paintable);
  gboolean has_crop;
  double tex_width;
  double tex_height;

  if (!self->gl_context)
    return;

  has_crop = has_effective_crop (self);
  tex_width = has_crop ? self->crop.width : self->format.info.raw.size.width;
  tex_height = has_crop ? self->crop.height : self->format.info.raw.size.height;

  if (self->paintable)
    {
      if (has_crop)
        {
          gtk_snapshot_save (snapshot);
          gtk_snapshot_translate (snapshot,
                                  &GRAPHENE_POINT_INIT (-self->crop.x, -self->crop.y));
          gtk_snapshot_scale (snapshot,
                              self->format.info.raw.size.width / self->crop.width,
                              self->format.info.raw.size.height / self->crop.height);

          gtk_snapshot_push_clip (snapshot,
                                  &GRAPHENE_RECT_INIT (0, 0, width, height));
          gdk_paintable_snapshot (self->paintable,
                                  snapshot,
                                  width,
                                  height);
          gtk_snapshot_pop (snapshot);

          gtk_snapshot_restore (snapshot);
        }
      else
        {
          gdk_paintable_snapshot (self->paintable,
                                  snapshot,
                                  width,
                                  height);
        }
    }

  if (self->cursor.valid && self->cursor.paintable)
    {
      double scale = MIN (width / tex_width, height / tex_height);
      double x_offset = self->cursor.x - self->cursor.hotspot_x;
      double y_offset = self->cursor.y - self->cursor.hotspot_y;

      gtk_snapshot_save (snapshot);
      gtk_snapshot_push_clip (snapshot,
                              &GRAPHENE_RECT_INIT (0, 0, width, height));
      gtk_snapshot_scale (snapshot, scale, scale);
      gtk_snapshot_translate (snapshot,
                              &GRAPHENE_POINT_INIT (x_offset, y_offset));

      gdk_paintable_snapshot (self->cursor.paintable,
                              snapshot,
                              self->cursor.width,
                              self->cursor.height);

      gtk_snapshot_pop (snapshot);
      gtk_snapshot_restore (snapshot);
    }
}

static GdkPaintable *
pw_media_stream_paintable_get_current_image (GdkPaintable *paintable)
{
  PwMediaStream *self = PW_MEDIA_STREAM (paintable);

  return self->paintable ? g_object_ref (self->paintable) : gdk_paintable_new_empty (0, 0);
}

static int
pw_media_stream_paintable_get_intrinsic_width (GdkPaintable *paintable)
{
  PwMediaStream *self = PW_MEDIA_STREAM (paintable);

  if (has_effective_crop (self))
    return self->crop.width;
  else
    return self->format.info.raw.size.width;
}

static int
pw_media_stream_paintable_get_intrinsic_height (GdkPaintable *paintable)
{
  PwMediaStream *self = PW_MEDIA_STREAM (paintable);

  if (has_effective_crop (self))
    return self->crop.height;
  else
    return self->format.info.raw.size.height;
}

static double
pw_media_stream_paintable_get_intrinsic_aspect_ratio (GdkPaintable *paintable)
{
  PwMediaStream *self = PW_MEDIA_STREAM (paintable);

  if (has_effective_crop (self))
    {
      return (double) self->crop.width / (double) self->crop.height;
    }
  else
    {
      return (double) self->format.info.raw.size.width /
             (double) self->format.info.raw.size.height;
    }
};

static void
pw_media_source_paintable_init (GdkPaintableInterface *iface)
{
  iface->snapshot = pw_media_stream_paintable_snapshot;
  iface->get_current_image = pw_media_stream_paintable_get_current_image;
  iface->get_intrinsic_width = pw_media_stream_paintable_get_intrinsic_width;
  iface->get_intrinsic_height = pw_media_stream_paintable_get_intrinsic_height;
  iface->get_intrinsic_aspect_ratio = pw_media_stream_paintable_get_intrinsic_aspect_ratio;
}


/*
 * GInitable interface
 */

static gboolean
pw_media_stream_initable_init (GInitable     *initable,
                               GCancellable  *cancellable,
                               GError       **error)
{
  PwMediaStream *self = PW_MEDIA_STREAM (initable);

  self->source = create_pipewire_source (self);
  if (!self->source)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED,
                   "Failed to create PipeWire source");
      return FALSE;
    }

  self->context = pw_context_new (self->source->pipewire_loop, NULL, 0);
  if (!self->source)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED,
                   "Failed to create pipewire context");
      return FALSE;
    }

  self->core = pw_context_connect_fd (self->context,
                                      fcntl (self->pipewire_fd, F_DUPFD_CLOEXEC, 5),
                                      NULL,
                                      0);
  if (!self->core)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED,
                   "Couldn't connect pipewire core");
      return FALSE;
    }

  pw_core_add_listener (self->core,
                        &self->core_listener,
                        &core_events,
                        self);

  self->renegotiate_event = pw_loop_add_event (self->source->pipewire_loop,
                                               renegotiate_stream_format,
                                               self);

  /* Fetch the server version */
  self->server_version_sync = pw_core_sync (self->core,
                                            PW_ID_CORE,
                                            self->server_version_sync);

  while (self->server_version_sync != -1)
    g_main_context_iteration (NULL, TRUE);

  /* Stream */
  self->stream = pw_stream_new (self->core,
                                "PwMediaStream",
                                pw_properties_new (PW_KEY_MEDIA_TYPE, "Video",
                                                   PW_KEY_MEDIA_CATEGORY, "Capture",
                                                   PW_KEY_MEDIA_ROLE, "Screen",
                                                   NULL));

  if (!self->stream)
    {
      g_set_error (error, G_IO_ERROR, G_IO_ERROR_FAILED,
                   "Couldn't connect PipeWire stream");
      return FALSE;
    }

  pw_stream_add_listener (self->stream,
                          &self->stream_listener,
                          &stream_events,
                          self);

  return TRUE;
}

static void
pw_media_source_initable_iface_init (GInitableIface *iface)
{
  iface->init = pw_media_stream_initable_init;
}


/*
 * GtkMediaStream overrides
 */

static gboolean
pw_media_stream_play (GtkMediaStream *media_stream)
{
  PwMediaStream *self = PW_MEDIA_STREAM (media_stream);

  if (self->stream)
    pw_stream_set_active (self->stream, TRUE);

  return self->stream != NULL;
}


static void
pw_media_stream_pause (GtkMediaStream *media_stream)
{
  PwMediaStream *self = PW_MEDIA_STREAM (media_stream);

  if (self->stream)
    pw_stream_set_active (self->stream, FALSE);
}

static void
pw_media_stream_realize (GtkMediaStream *media_stream,
                         GdkSurface     *surface)
{
  PwMediaStream *self = PW_MEDIA_STREAM (media_stream);
  g_autoptr (GError) error = NULL;

  if (self->gl_context)
    return;

  self->gl_context = gdk_surface_create_gl_context (surface, &error);
  if (error)
    {
      g_critical ("Failed to create GDK GL context: %s", error->message);
      return;
    }

  gdk_gl_context_realize (self->gl_context, &error);
  if (error)
    {
      g_critical ("Failed to realize GDK GL context: %s", error->message);
      g_clear_object (&self->gl_context);
      return;
    }

  query_formats_and_modifiers (self);

  if (self->node_id != PW_ID_ANY)
    connect_stream (self);
  else
    select_node (self);
}

static void
pw_media_stream_unrealize (GtkMediaStream *media_stream,
                           GdkSurface     *surface)
{
  PwMediaStream *self = PW_MEDIA_STREAM (media_stream);

  if (!self->gl_context)
    return;

  if (gdk_gl_context_get_surface (self->gl_context) == surface)
    g_clear_object (&self->gl_context);
}


/*
 * GObject overrides
 */

static void
pw_media_stream_dispose (GObject *object)
{
  PwMediaStream *self = (PwMediaStream *)object;

  g_clear_object (&self->paintable);
  g_clear_object (&self->cursor.paintable);
  g_clear_object (&self->gl_context);

  g_clear_pointer (&self->formats, g_array_unref);

  if (self->stream)
    pw_stream_disconnect (self->stream);
  g_clear_pointer (&self->stream, pw_stream_destroy);
  g_clear_pointer (&self->context, pw_context_destroy);
  g_clear_pointer ((GSource**)&self->source, g_source_destroy);

  if (self->pipewire_fd > 0)
    {
      close (self->pipewire_fd);
      self->pipewire_fd = 0;
    }

  G_OBJECT_CLASS (pw_media_stream_parent_class)->dispose (object);
}

static void
pw_media_stream_get_property (GObject    *object,
                              guint       prop_id,
                              GValue     *value,
                              GParamSpec *pspec)
{
  PwMediaStream *self = PW_MEDIA_STREAM (object);

  switch (prop_id)
    {
    case PROP_PIPEWIRE_FD:
      g_value_set_int (value, self->pipewire_fd);
      break;

    case PROP_NODE_ID:
      g_value_set_uint (value, self->node_id);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
pw_media_stream_set_property (GObject      *object,
                              guint         prop_id,
                              const GValue *value,
                              GParamSpec   *pspec)
{
  PwMediaStream *self = PW_MEDIA_STREAM (object);

  switch (prop_id)
    {
    case PROP_PIPEWIRE_FD:
      g_assert (self->pipewire_fd == 0);
      self->pipewire_fd = g_value_get_int (value);
      break;

    case PROP_NODE_ID:
      g_assert (self->node_id == 0);
      self->node_id = g_value_get_uint (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
pw_media_stream_class_init (PwMediaStreamClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkMediaStreamClass *media_stream_class = GTK_MEDIA_STREAM_CLASS (klass);

  object_class->dispose = pw_media_stream_dispose;
  object_class->get_property = pw_media_stream_get_property;
  object_class->set_property = pw_media_stream_set_property;

  media_stream_class->play = pw_media_stream_play;
  media_stream_class->pause = pw_media_stream_pause;
  media_stream_class->realize = pw_media_stream_realize;
  media_stream_class->unrealize = pw_media_stream_unrealize;

  properties[PROP_PIPEWIRE_FD] =
    g_param_spec_int ("pipewire-fd", "", "",
                      0, G_MAXINT, 0,
                      G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS);

  properties[PROP_NODE_ID] =
    g_param_spec_uint ("node-id", "", "",
                       0, G_MAXUINT, 0,
                       G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS);

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
pw_media_stream_init (PwMediaStream *self)
{
}

GtkMediaStream *
pw_media_stream_new (int        pipewire_fd,
                     uint32_t   node_id,
                     GError   **error)
{
  return g_initable_new (PW_TYPE_MEDIA_STREAM,
                         NULL,
                         error,
                         "pipewire-fd", pipewire_fd,
                         "node-id", node_id,
                         NULL);
}
