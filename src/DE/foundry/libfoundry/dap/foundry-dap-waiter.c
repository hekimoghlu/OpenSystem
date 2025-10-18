/* foundry-dap-waiter.c
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

#include "foundry-dap-waiter-private.h"

struct _FoundryDapWaiter
{
  GObject     parent_instance;
  DexPromise *promise;
  JsonNode   *node;
  gint64      seq;
};

G_DEFINE_FINAL_TYPE (FoundryDapWaiter, foundry_dap_waiter, G_TYPE_OBJECT)

static void
foundry_dap_waiter_finalize (GObject *object)
{
  FoundryDapWaiter *self = (FoundryDapWaiter *)object;

  g_clear_pointer (&self->node, json_node_unref);

  if (dex_future_is_pending (DEX_FUTURE (self->promise)))
    dex_promise_reject (self->promise,
                        g_error_new_literal (G_IO_ERROR,
                                             G_IO_ERROR_TIMED_OUT,
                                             "Timed out"));

  dex_clear (&self->promise);

  G_OBJECT_CLASS (foundry_dap_waiter_parent_class)->finalize (object);
}

static void
foundry_dap_waiter_class_init (FoundryDapWaiterClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_dap_waiter_finalize;
}

static void
foundry_dap_waiter_init (FoundryDapWaiter *self)
{
  self->promise = dex_promise_new ();
}

FoundryDapWaiter *
foundry_dap_waiter_new (JsonNode *node,
                        gint64    seq)
{
  FoundryDapWaiter *self;

  g_return_val_if_fail (node != NULL, NULL);

  self = g_object_new (FOUNDRY_TYPE_DAP_WAITER, NULL);
  self->node = json_node_ref (node);
  self->seq = seq;

  return self;
}

/**
 * foundry_dap_waiter_await:
 * @self: a [class@Foundry.DapWaiter]
 *
 * Returns: (transfer full);
 */
DexFuture *
foundry_dap_waiter_await (FoundryDapWaiter *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DAP_WAITER (self));

  return dex_ref (DEX_FUTURE (self->promise));
}

/**
 * foundry_dap_waiter_reply:
 * @self: a [class@Foundry.DapWaiter]
 * @node: (transfer full) (nullable): the reply node
 *
 */
void
foundry_dap_waiter_reply (FoundryDapWaiter *self,
                          JsonNode         *node)
{
  g_return_if_fail (FOUNDRY_IS_DAP_WAITER (self));

  if (dex_future_is_pending (DEX_FUTURE (self->promise)))
    dex_promise_resolve_boxed (self->promise, JSON_TYPE_NODE, g_steal_pointer (&node));
}

/**
 * foundry_dap_waiter_reject:
 * @self: a [class@Foundry.DapWaiter]
 * @error: (transfer full):
 *
 * Fails the awaiting future, using @error
 */
void
foundry_dap_waiter_reject (FoundryDapWaiter *self,
                           GError           *error)
{
  g_return_if_fail (FOUNDRY_IS_DAP_WAITER (self));
  g_return_if_fail (error != NULL);

  if (!dex_future_is_pending (DEX_FUTURE (self->promise)))
    dex_promise_reject (self->promise, g_steal_pointer (&error));
}

DexFuture *
foundry_dap_waiter_catch (DexFuture *completed,
                          gpointer   user_data)
{
  FoundryDapWaiter *self = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_DAP_WAITER (self));

  if (!dex_await (dex_ref (completed), &error))
    foundry_dap_waiter_reject (self, g_steal_pointer (&error));

  return dex_ref (completed);
}

/**
 * foundry_dap_waiter_get_node:
 *
 * Returns: (transfer none): a node
 */
JsonNode *
foundry_dap_waiter_get_node (FoundryDapWaiter *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_WAITER (self), NULL);

  return self->node;
}

/**
 * foundry_dap_waiter_get_seq:
 * @self: a [class@Foundry.DapWaiter]
 */
gint64
foundry_dap_waiter_get_seq (FoundryDapWaiter *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DAP_WAITER (self), 0);

  return self->seq;
}
