/*
 * mks-screen-resizer.c
 *
 * Copyright 2023 Bilal Elmoussaoui <belmouss@redhat.com>
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

#include "mks-screen-attributes.h"
#include "mks-screen-resizer-private.h"
#include "mks-util-private.h"

static void mks_screen_resizer_reconfigure (MksScreenResizer    *self,
                                            MksScreenAttributes *attributes);

struct _MksScreenResizer
{
  GObject              parent_instance;

  MksScreen           *screen;

  /* Remember our last operation */
  MksScreenAttributes *next_op;
  MksScreenAttributes *previous_op;

  guint                in_progress : 1;
};

enum {
  PROP_0,
  PROP_SCREEN,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (MksScreenResizer, mks_screen_resizer, G_TYPE_OBJECT)

static GParamSpec *properties [N_PROPS];

static void
mks_screen_resizer_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  MksScreenResizer *self = MKS_SCREEN_RESIZER (object);

  switch (prop_id)
    {
    case PROP_SCREEN:
      g_value_set_object (value, self->screen);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_screen_resizer_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  MksScreenResizer *self = MKS_SCREEN_RESIZER (object);

  switch (prop_id)
    {
    case PROP_SCREEN:
      mks_screen_resizer_set_screen (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
mks_screen_resizer_dispose (GObject *object)
{
  MksScreenResizer *self = (MksScreenResizer *)object;

  g_clear_object (&self->screen);
  g_clear_pointer (&self->next_op, mks_screen_attributes_free);
  g_clear_pointer (&self->previous_op, mks_screen_attributes_free);

  G_OBJECT_CLASS (mks_screen_resizer_parent_class)->dispose (object);
}

static void
mks_screen_resizer_class_init (MksScreenResizerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = mks_screen_resizer_dispose;
  object_class->get_property = mks_screen_resizer_get_property;
  object_class->set_property = mks_screen_resizer_set_property;

  properties[PROP_SCREEN] =
    g_param_spec_object ("screen", NULL, NULL,
                         MKS_TYPE_SCREEN,
                         (G_PARAM_READWRITE | G_PARAM_EXPLICIT_NOTIFY | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
on_screen_configure_cb (GObject      *object,
                        GAsyncResult *result,
                        gpointer      user_data)
{
  MksScreen *screen = (MksScreen *)object;
  g_autoptr(MksScreenAttributes) attributes = NULL;
  g_autoptr(MksScreenResizer) self = user_data;
  g_autoptr(GError) error = NULL;

  MKS_ENTRY;

  g_assert (MKS_IS_SCREEN (screen));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (MKS_IS_SCREEN_RESIZER (self));

  if (!mks_screen_configure_finish (screen, result, &error))
    g_debug ("Screen configure failed: %s", error->message);

  self->in_progress = FALSE;
  attributes = g_steal_pointer (&self->next_op);

  if (attributes && !mks_screen_attributes_equal (attributes, self->previous_op))
    mks_screen_resizer_reconfigure (self, g_steal_pointer (&attributes));

  MKS_EXIT;
}

/**
 * mks_screen_resizer_new:
 *
 * Returns: (transfer full): a new #MksScreenResizer
 */
MksScreenResizer *
mks_screen_resizer_new (void)
{
  return g_object_new (MKS_TYPE_SCREEN_RESIZER, NULL);
}


static void
mks_screen_resizer_init (MksScreenResizer *self)
{
}

/**
 * mks_screen_resizer_set_screen:
 * @self: A `MksScreenResizer`
 * @screen: A `MksScreen`
 *
 * Sets the screen to resize when a resize is queued.
*/
void
mks_screen_resizer_set_screen (MksScreenResizer *self,
                               MksScreen        *screen)
{
  MKS_ENTRY;

  g_return_if_fail (MKS_IS_SCREEN_RESIZER (self));
  g_return_if_fail (!screen || MKS_IS_SCREEN (screen));

  if (g_set_object (&self->screen, screen))
    g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_SCREEN]);

  MKS_EXIT;
}

/**
 * mks_screen_resizer_queue_resize:
 * @self: A `MksScreenResizer`
 * @attributes: (transfer full): The new attributes to queue
 *
 * Schedule the VM display configuration with the passed attributes if
 * there is no ongoing operation. Otherwise, add the attributes to
 * a queue of updates.
*/
void
mks_screen_resizer_queue_resize (MksScreenResizer    *self,
                                 MksScreenAttributes *attributes)
{
  MKS_ENTRY;

  g_return_if_fail (MKS_IS_SCREEN_RESIZER (self));

  if (mks_screen_attributes_equal (attributes, self->previous_op))
    MKS_EXIT;

  if (self->in_progress)
    {
      g_clear_pointer (&self->next_op, mks_screen_attributes_free);
      self->next_op = g_steal_pointer (&attributes);
      MKS_EXIT;
    }

  mks_screen_resizer_reconfigure (self, attributes);

  MKS_EXIT;
}

/**
 * mks_screen_resizer_reconfigure:
 * @self: A `MksScreenResizer`
 * @attributes: (transfer full): The attributes to reconfigure
 *
 * Configure the screen with the passed attributes.
*/
static void
mks_screen_resizer_reconfigure (MksScreenResizer    *self,
                                MksScreenAttributes *attributes)
{
  MKS_ENTRY;

  g_assert (MKS_IS_SCREEN_RESIZER (self));

  self->in_progress = TRUE;

  g_clear_pointer (&self->previous_op, mks_screen_attributes_free);
  self->previous_op = mks_screen_attributes_copy (attributes);

  mks_screen_configure (self->screen,
                        g_steal_pointer (&attributes),
                        NULL,
                        on_screen_configure_cb,
                        g_object_ref (self));

  MKS_EXIT;
}
