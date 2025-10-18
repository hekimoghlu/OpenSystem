/* plugin-devhelp-progress.c
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

#include "plugin-devhelp-progress.h"

struct _PluginDevhelpProgress
{
  GObject parent_instance;
  GPtrArray *jobs;
  guint removed;
  guint done : 1;
};

enum {
  PROP_0,
  PROP_FRACTION,
  PROP_N_ITEMS,
  N_PROPS
};

enum {
  OP_ADDED = 0,
  OP_REMOVED = 1,
  OP_FRACTION = 2,
  N_OPS
};

static guint
plugin_devhelp_progress_get_n_items (GListModel *model)
{
  return PLUGIN_DEVHELP_PROGRESS (model)->jobs->len;
}

static GType
plugin_devhelp_progress_get_item_type (GListModel *model)
{
  return PLUGIN_TYPE_DEVHELP_JOB;
}

static gpointer
plugin_devhelp_progress_get_item (GListModel *model,
                                  guint       position)
{
  PluginDevhelpProgress *self = PLUGIN_DEVHELP_PROGRESS (model);

  if (position < self->jobs->len)
    return g_object_ref (g_ptr_array_index (self->jobs, position));

  return NULL;
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_n_items = plugin_devhelp_progress_get_n_items;
  iface->get_item_type = plugin_devhelp_progress_get_item_type;
  iface->get_item = plugin_devhelp_progress_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (PluginDevhelpProgress, plugin_devhelp_progress, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static GParamSpec *properties[N_PROPS];

typedef struct _NotifyInMain
{
  PluginDevhelpProgress *self;
  PluginDevhelpJob *job;
  guint op : 2;
} NotifyInMain;

static void
notify_in_main_cb (gpointer user_data)
{
  NotifyInMain *state = user_data;

  g_assert (state != NULL);
  g_assert (PLUGIN_IS_DEVHELP_PROGRESS (state->self));
  g_assert (PLUGIN_IS_DEVHELP_JOB (state->job));

  if (state->op == OP_ADDED)
    {
      g_ptr_array_add (state->self->jobs, g_object_ref (state->job));
      g_list_model_items_changed (G_LIST_MODEL (state->self),
                                  state->self->jobs->len - 1,
                                  0, 1);
      g_object_notify_by_pspec (G_OBJECT (state->self), properties[PROP_N_ITEMS]);
    }
  else if (state->op == OP_REMOVED)
    {
      guint pos = 0;

      if (g_ptr_array_find (state->self->jobs, state->job, &pos))
        {
          state->self->removed++;
          g_ptr_array_remove_index (state->self->jobs, pos);
          g_list_model_items_changed (G_LIST_MODEL (state->self), pos, 1, 0);
          g_object_notify_by_pspec (G_OBJECT (state->self), properties[PROP_N_ITEMS]);
        }
    }
  else if (state->op == OP_FRACTION) { /* Do nothing */ }

  g_object_notify_by_pspec (G_OBJECT (state->self),
                            properties[PROP_FRACTION]);

  g_clear_object (&state->self);
  g_clear_object (&state->job);
  g_free (state);
}

static void
notify_in_main (PluginDevhelpProgress *self,
                PluginDevhelpJob      *job,
                guint                  op)
{
  NotifyInMain *state;

  g_assert (PLUGIN_IS_DEVHELP_PROGRESS (self));
  g_assert (PLUGIN_IS_DEVHELP_JOB (job));
  g_assert (op < N_OPS);

  state = g_new0 (NotifyInMain, 1);
  state->self = g_object_ref (self);
  state->job = g_object_ref (job);
  state->op = op;

  dex_scheduler_push (dex_scheduler_get_default (),
                      notify_in_main_cb,
                      state);
}

static void
plugin_devhelp_progress_finalize (GObject *object)
{
  PluginDevhelpProgress *self = (PluginDevhelpProgress *)object;

  g_clear_pointer (&self->jobs, g_ptr_array_unref);

  G_OBJECT_CLASS (plugin_devhelp_progress_parent_class)->finalize (object);
}

static void
plugin_devhelp_progress_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  PluginDevhelpProgress *self = PLUGIN_DEVHELP_PROGRESS (object);

  switch (prop_id)
    {
    case PROP_FRACTION:
      g_value_set_double (value, plugin_devhelp_progress_get_fraction (self));
      break;

    case PROP_N_ITEMS:
      g_value_set_uint (value, self->jobs->len);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_devhelp_progress_class_init (PluginDevhelpProgressClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = plugin_devhelp_progress_finalize;
  object_class->get_property = plugin_devhelp_progress_get_property;

  properties[PROP_FRACTION] =
    g_param_spec_double ("fraction", NULL, NULL,
                         0, 1, 0,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_N_ITEMS] =
    g_param_spec_uint ("n-items", NULL, NULL,
                       0, G_MAXUINT-1, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_devhelp_progress_init (PluginDevhelpProgress *self)
{
  self->jobs = g_ptr_array_new_with_free_func (g_object_unref);
}

PluginDevhelpProgress *
plugin_devhelp_progress_new (void)
{
  return g_object_new (PLUGIN_TYPE_DEVHELP_PROGRESS, NULL);
}

static void
plugin_devhelp_progress_job_completed_cb (PluginDevhelpProgress *self,
                                          PluginDevhelpJob      *job)
{
  g_assert (PLUGIN_IS_DEVHELP_PROGRESS (self));
  g_assert (PLUGIN_IS_DEVHELP_JOB (job));

  notify_in_main (self, job, OP_REMOVED);
}

static void
plugin_devhelp_progress_job_notify_fraction_cb (PluginDevhelpProgress *self,
                                                GParamSpec            *pspec,
                                                PluginDevhelpJob      *job)
{
  g_assert (PLUGIN_IS_DEVHELP_PROGRESS (self));
  g_assert (PLUGIN_IS_DEVHELP_JOB (job));

  notify_in_main (self, job, OP_FRACTION);
}

PluginDevhelpJob *
plugin_devhelp_progress_begin_job (PluginDevhelpProgress *self)
{
  PluginDevhelpJob *job;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_PROGRESS (self), NULL);

  job = g_object_new (PLUGIN_TYPE_DEVHELP_JOB, NULL);
  g_signal_connect_object (job,
                           "notify::fraction",
                           G_CALLBACK (plugin_devhelp_progress_job_notify_fraction_cb),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (job,
                           "completed",
                           G_CALLBACK (plugin_devhelp_progress_job_completed_cb),
                           self,
                           G_CONNECT_SWAPPED);
  notify_in_main (self, job, OP_ADDED);
  return job;
}

double
plugin_devhelp_progress_get_fraction (PluginDevhelpProgress *self)
{
  double total = 0;
  double numerator;
  double denominator;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_PROGRESS (self), 0);

  if (self->done)
    return 1;

  if (self->jobs->len == 0)
    return 0;

  for (guint i = 0; i < self->jobs->len; i++)
    {
      PluginDevhelpJob *job = g_ptr_array_index (self->jobs, i);
      total += CLAMP (plugin_devhelp_job_get_fraction (job), 0, 1);
    }

  numerator = total + self->removed;
  denominator = self->jobs->len + self->removed;

  return numerator / denominator;
}

void
plugin_devhelp_progress_done (PluginDevhelpProgress *self)
{
  g_return_if_fail (PLUGIN_IS_DEVHELP_PROGRESS (self));

  self->done = TRUE;
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_FRACTION]);
}
