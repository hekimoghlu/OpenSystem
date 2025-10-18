/* foundry-log-model.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-debug.h"
#include "foundry-log-model-private.h"
#include "foundry-log-message-private.h"

struct _FoundryLogModel
{
  GObject     parent_instance;
  DexChannel *channel;
  GSequence  *items;
  guint       n_items;
};

enum {
  PROP_0,
  PROP_N_ITEMS,
  N_PROPS
};

static GType
foundry_log_model_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_LOG_MESSAGE;
}

static guint
foundry_log_model_get_n_items (GListModel *model)
{
  return FOUNDRY_LOG_MODEL (model)->n_items;
}

static gpointer
foundry_log_model_get_item (GListModel *model,
                            guint       position)
{
  FoundryLogModel *self = FOUNDRY_LOG_MODEL (model);

  g_assert (FOUNDRY_IS_MAIN_THREAD ());

  if (position < self->n_items)
    return g_object_ref (g_sequence_get (g_sequence_get_iter_at_pos (self->items, position)));

  return NULL;
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item = foundry_log_model_get_item;
  iface->get_n_items = foundry_log_model_get_n_items;
  iface->get_item_type = foundry_log_model_get_item_type;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryLogModel, foundry_log_model, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static GParamSpec *properties [N_PROPS];

typedef struct
{
  GWeakRef    model_wr;
  DexChannel *channel;
} ChannelData;

static ChannelData *
channel_data_new (FoundryLogModel *self)
{
  ChannelData *cd;

  cd = g_new0 (ChannelData, 1);
  g_weak_ref_init (&cd->model_wr, self);
  cd->channel = dex_ref (self->channel);

  return cd;
}

static void
channel_data_free (ChannelData *cd)
{
  g_weak_ref_clear (&cd->model_wr);
  dex_clear (&cd->channel);
  g_free (cd);
}

static void
foundry_log_model_take (FoundryLogModel   *self,
                        FoundryLogMessage *item)
{
  guint position;

  g_assert (FOUNDRY_IS_LOG_MODEL (self));
  g_assert (FOUNDRY_IS_LOG_MESSAGE (item));

  /* We only want to emit log messages from the main thread so that
   * consumers do not need to manually proxy messages back to the
   * main thread on our behalf.
   */
  if G_UNLIKELY (!FOUNDRY_IS_MAIN_THREAD ())
    {
      g_autoptr(DexFuture) future = NULL;

      future = dex_channel_send (self->channel,
                                 dex_future_new_take_object (item));
      return;
    }

  position = self->n_items, self->n_items++;
  g_sequence_append (self->items, g_steal_pointer (&item));
  g_list_model_items_changed (G_LIST_MODEL (self), position, 0, 1);
  g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_N_ITEMS]);
}

static DexFuture *
foundry_log_model_fiber_func (gpointer data)
{
  ChannelData *cd = data;
  FoundryLogMessage *item;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (cd != NULL);
  g_assert (DEX_IS_CHANNEL (cd->channel));

  while ((item = dex_await_object (dex_channel_receive (cd->channel), NULL)))
    {
      g_autoptr(FoundryLogModel) self = g_weak_ref_get (&cd->model_wr);

      if (self != NULL)
        foundry_log_model_take (self, g_steal_pointer (&item));
      else
        g_object_unref (g_steal_pointer (&item));
    }

  return NULL;
}

static void
foundry_log_model_dispose (GObject *object)
{
  FoundryLogModel *self = (FoundryLogModel *)object;

  if (self->n_items > 0)
    {
      self->n_items = 0;
      g_sequence_remove_range (g_sequence_get_begin_iter (self->items),
                               g_sequence_get_end_iter (self->items));
    }

  if (dex_channel_can_send (self->channel))
    dex_channel_close_send (self->channel);

  G_OBJECT_CLASS (foundry_log_model_parent_class)->dispose (object);
}

static void
foundry_log_model_finalize (GObject *object)
{
  FoundryLogModel *self = (FoundryLogModel *)object;

  g_clear_pointer (&self->items, g_sequence_free);
  dex_clear (&self->channel);

  G_OBJECT_CLASS (foundry_log_model_parent_class)->finalize (object);
}

static void
foundry_log_model_get_property (GObject    *object,
                                guint       prop_id,
                                GValue     *value,
                                GParamSpec *pspec)
{
  FoundryLogModel *self = FOUNDRY_LOG_MODEL (object);

  switch (prop_id)
    {
    case PROP_N_ITEMS:
      g_value_set_uint (value, self->n_items);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_log_model_class_init (FoundryLogModelClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_log_model_dispose;
  object_class->finalize = foundry_log_model_finalize;
  object_class->get_property = foundry_log_model_get_property;

  properties[PROP_N_ITEMS] =
    g_param_spec_uint ("n-items", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_log_model_init (FoundryLogModel *self)
{
  self->items = g_sequence_new (g_object_unref);
  self->channel = dex_channel_new (0);

  dex_future_disown (dex_scheduler_spawn (NULL,
                                          0,
                                          foundry_log_model_fiber_func,
                                          channel_data_new (self),
                                          (GDestroyNotify)channel_data_free));
}

FoundryLogModel *
_foundry_log_model_new (void)
{
  return g_object_new (FOUNDRY_TYPE_LOG_MODEL, NULL);
}

void
_foundry_log_model_append (FoundryLogModel *self,
                           const char      *domain,
                           GLogLevelFlags   flags,
                           char            *message)
{
  foundry_log_model_take (self, _foundry_log_message_new (flags, domain, message, NULL));
}


void
_foundry_log_model_remove_all (FoundryLogModel *self)
{
  guint n_items;

  g_return_if_fail (FOUNDRY_IS_LOG_MODEL (self));

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self));
  g_clear_pointer (&self->items, g_sequence_free);
  self->items = g_sequence_new (g_object_unref);
  self->n_items = 0;

  if (n_items > 0)
    {
      g_list_model_items_changed (G_LIST_MODEL (self), 0, n_items, 0);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_N_ITEMS]);
    }
}
