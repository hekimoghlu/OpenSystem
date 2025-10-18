/* plugin-devhelp-job.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <libdex.h>

#include "plugin-devhelp-job.h"

struct _PluginDevhelpJob
{
  GObject parent_instance;
  GMutex mutex;
  char *title;
  char *subtitle;
  double fraction;
  guint has_completed : 1;
};

enum {
  PROP_0,
  PROP_TITLE,
  PROP_SUBTITLE,
  PROP_FRACTION,
  N_PROPS
};

enum {
  COMPLETED,
  N_SIGNALS
};

G_DEFINE_FINAL_TYPE (PluginDevhelpJob, plugin_devhelp_job, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];
static guint signals[N_SIGNALS];

typedef struct _NotifyInMain
{
  PluginDevhelpJob *self;
  GParamSpec *pspec;
} NotifyInMain;

static void
notify_in_main_cb (gpointer user_data)
{
  NotifyInMain *state = user_data;

  g_assert (state != NULL);
  g_assert (PLUGIN_IS_DEVHELP_JOB (state->self));
  g_assert (state->pspec != NULL);

  g_object_notify_by_pspec (G_OBJECT (state->self), state->pspec);

  g_clear_object (&state->self);
  g_clear_pointer (&state->pspec, g_param_spec_unref);
  g_free (state);
}

static void
notify_in_main (PluginDevhelpJob *self,
                GParamSpec       *pspec)
{
  NotifyInMain *state;

  g_assert (PLUGIN_IS_DEVHELP_JOB (self));
  g_assert (pspec != NULL);

  state = g_new0 (NotifyInMain, 1);
  state->self = g_object_ref (self);
  state->pspec = g_param_spec_ref (pspec);

  dex_scheduler_push (dex_scheduler_get_default (),
                      notify_in_main_cb,
                      state);
}

static void
plugin_devhelp_job_finalize (GObject *object)
{
  PluginDevhelpJob *self = (PluginDevhelpJob *)object;

  g_clear_pointer (&self->title, g_free);
  g_clear_pointer (&self->subtitle, g_free);
  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (plugin_devhelp_job_parent_class)->finalize (object);
}

static void
plugin_devhelp_job_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  PluginDevhelpJob *self = PLUGIN_DEVHELP_JOB (object);

  switch (prop_id)
    {
    case PROP_FRACTION:
      g_value_set_double (value, plugin_devhelp_job_get_fraction (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, plugin_devhelp_job_dup_title (self));
      break;

    case PROP_SUBTITLE:
      g_value_take_string (value, plugin_devhelp_job_dup_subtitle (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_devhelp_job_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  PluginDevhelpJob *self = PLUGIN_DEVHELP_JOB (object);

  switch (prop_id)
    {
    case PROP_FRACTION:
      plugin_devhelp_job_set_fraction (self, g_value_get_double (value));
      break;

    case PROP_TITLE:
      plugin_devhelp_job_set_title (self, g_value_get_string (value));
      break;

    case PROP_SUBTITLE:
      plugin_devhelp_job_set_subtitle (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_devhelp_job_class_init (PluginDevhelpJobClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = plugin_devhelp_job_finalize;
  object_class->get_property = plugin_devhelp_job_get_property;
  object_class->set_property = plugin_devhelp_job_set_property;

  signals[COMPLETED] =
    g_signal_new ("completed",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 0);

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SUBTITLE] =
    g_param_spec_string ("subtitle", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_FRACTION] =
    g_param_spec_double ("fraction", NULL, NULL,
                         0, 1, 0,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_devhelp_job_init (PluginDevhelpJob *self)
{
  g_mutex_init (&self->mutex);
}

PluginDevhelpJob *
plugin_devhelp_job_new (void)
{
  return g_object_new (PLUGIN_TYPE_DEVHELP_JOB, NULL);
}

char *
plugin_devhelp_job_dup_title (PluginDevhelpJob *self)
{
  char *ret;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_JOB (self), NULL);

  g_mutex_lock (&self->mutex);
  ret = g_strdup (self->title);
  g_mutex_unlock (&self->mutex);

  return ret;
}

void
plugin_devhelp_job_set_title (PluginDevhelpJob *self,
                              const char       *title)
{
  g_return_if_fail (PLUGIN_IS_DEVHELP_JOB (self));

  g_mutex_lock (&self->mutex);
  if (g_set_str (&self->title, title))
    notify_in_main (self, properties[PROP_TITLE]);
  g_mutex_unlock (&self->mutex);
}

char *
plugin_devhelp_job_dup_subtitle (PluginDevhelpJob *self)
{
  char *ret;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_JOB (self), NULL);

  g_mutex_lock (&self->mutex);
  ret = g_strdup (self->subtitle);
  g_mutex_unlock (&self->mutex);

  return ret;
}

void
plugin_devhelp_job_set_subtitle (PluginDevhelpJob *self,
                                 const char       *subtitle)
{
  g_return_if_fail (PLUGIN_IS_DEVHELP_JOB (self));

  g_mutex_lock (&self->mutex);
  if (g_set_str (&self->subtitle, subtitle))
    notify_in_main (self, properties[PROP_SUBTITLE]);
  g_mutex_unlock (&self->mutex);
}

double
plugin_devhelp_job_get_fraction (PluginDevhelpJob *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_JOB (self), 0);

  return self->fraction;
}

void
plugin_devhelp_job_set_fraction (PluginDevhelpJob *self,
                                 double            fraction)
{
  g_return_if_fail (PLUGIN_IS_DEVHELP_JOB (self));

  g_mutex_lock (&self->mutex);
  if (fraction != self->fraction)
    {
      self->fraction = fraction;
      notify_in_main (self, properties[PROP_FRACTION]);
    }
  g_mutex_unlock (&self->mutex);
}

void
plugin_devhelp_job_complete (PluginDevhelpJob *self)
{
  gboolean can_emit;

  g_return_if_fail (PLUGIN_IS_DEVHELP_JOB (self));

  g_mutex_lock (&self->mutex);
  can_emit = self->has_completed == FALSE;
  self->has_completed = TRUE;
  self->fraction = 1;
  g_mutex_unlock (&self->mutex);

  if (can_emit)
    {
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_FRACTION]);
      g_signal_emit (self, signals[COMPLETED], 0);
    }
}
