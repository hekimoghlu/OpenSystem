/* foundry-service.c
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

#include "foundry-action-muxer.h"
#include "foundry-service-private.h"

typedef struct
{
  DexPromise *started;
  DexPromise *stopped;
  guint       has_started : 1;
  guint       has_stopped : 1;
} FoundryServicePrivate;

typedef struct
{
  const char         *action_prefix;
  FoundryActionMixin  actions;
} FoundryServiceClassPrivate;

G_DEFINE_ABSTRACT_TYPE_WITH_CODE (FoundryService, foundry_service, FOUNDRY_TYPE_CONTEXTUAL,
                                  G_ADD_PRIVATE (FoundryService)
                                  g_type_add_class_private (g_define_type_id, sizeof (FoundryServiceClassPrivate));)

G_DEFINE_QUARK (foundry_service_error, foundry_service_error)

static inline FoundryServiceClassPrivate *
foundry_service_class_get_private (FoundryServiceClass *klass)
{
  return g_type_class_get_private ((GTypeClass *)klass, FOUNDRY_TYPE_SERVICE);
}

static DexFuture *
foundry_service_real_start (FoundryService *self)
{
  return dex_future_new_true ();
}

static DexFuture *
foundry_service_real_stop (FoundryService *self)
{
  return dex_future_new_true ();
}

static void
foundry_service_constructed (GObject *object)
{
  FoundryService *self = (FoundryService *)object;
  FoundryServiceClass *klass = FOUNDRY_SERVICE_GET_CLASS (self);
  FoundryServiceClassPrivate *klass_priv = foundry_service_class_get_private (klass);

  G_OBJECT_CLASS (foundry_service_parent_class)->constructed (object);

  foundry_action_mixin_constructed (&klass_priv->actions, self);
}

static void
foundry_service_finalize (GObject *object)
{
  FoundryService *self = (FoundryService *)object;
  FoundryServicePrivate *priv = foundry_service_get_instance_private (self);

  dex_clear (&priv->started);
  dex_clear (&priv->stopped);

  G_OBJECT_CLASS (foundry_service_parent_class)->finalize (object);
}

static void
foundry_service_class_init (FoundryServiceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryServiceClassPrivate *priv = foundry_service_class_get_private (klass);

  object_class->constructed = foundry_service_constructed;
  object_class->finalize = foundry_service_finalize;

  klass->start = foundry_service_real_start;
  klass->stop = foundry_service_real_stop;

  foundry_action_mixin_init (&priv->actions, object_class);
}

static void
foundry_service_init (FoundryService *self)
{
  FoundryServicePrivate *priv = foundry_service_get_instance_private (self);

  priv->started = dex_promise_new ();
  priv->stopped = dex_promise_new ();
}

/**
 * foundry_service_when_ready:
 * @self: a #FoundryService
 *
 * Gets a future that resolves when the service has started.
 *
 * Returns: (transfer full): a #DexFuture
 */
DexFuture *
foundry_service_when_ready (FoundryService *self)
{
  FoundryServicePrivate *priv = foundry_service_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_SERVICE (self), NULL);

  if (priv->has_stopped)
    return dex_future_new_reject (FOUNDRY_SERVICE_ERROR,
                                  FOUNDRY_SERVICE_ERROR_ALREADY_STOPPED,
                                  "Service has already been shutdown");

  return dex_ref (priv->started);
}

/**
 * foundry_service_when_shutdown:
 * @self: a #FoundryService
 *
 * Gets a future that resolves when the service has shutdown.
 *
 * Returns: (transfer full): A #DexFuture
 */
DexFuture *
foundry_service_when_shutdown (FoundryService *self)
{
  FoundryServicePrivate *priv = foundry_service_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_SERVICE (self), NULL);

  return dex_ref (priv->stopped);
}

static DexFuture *
foundry_service_propagate (DexFuture *from,
                           gpointer   user_data)
{
  DexPromise *to = user_data;
  g_autoptr(GError) error = NULL;
  const GValue *value;

  g_assert (DEX_IS_FUTURE (from));
  g_assert (DEX_IS_PROMISE (to));

  if ((value = dex_future_get_value (from, &error)))
    dex_promise_resolve (to, value);
  else
    dex_promise_reject (to, g_steal_pointer (&error));

  return NULL;
}

DexFuture *
foundry_service_start (FoundryService *self)
{
  FoundryServicePrivate *priv = foundry_service_get_instance_private (self);
  DexFuture *future;

  g_return_val_if_fail (FOUNDRY_IS_SERVICE (self), NULL);

  if (priv->has_started)
    return dex_future_new_reject (FOUNDRY_SERVICE_ERROR,
                                  FOUNDRY_SERVICE_ERROR_ALREADY_STARTED,
                                  "Service already started");

  priv->has_started = TRUE;

  g_debug ("Starting service %s", G_OBJECT_TYPE_NAME (self));

  future = FOUNDRY_SERVICE_GET_CLASS (self)->start (self);
  future = dex_future_finally (future,
                               foundry_service_propagate,
                               dex_ref (priv->started),
                               dex_unref);

  return future;
}

DexFuture *
foundry_service_stop (FoundryService *self)
{
  FoundryServicePrivate *priv = foundry_service_get_instance_private (self);
  DexFuture *future;

  g_return_val_if_fail (FOUNDRY_IS_SERVICE (self), NULL);

  if (priv->has_stopped)
    return dex_future_new_reject (FOUNDRY_SERVICE_ERROR,
                                  FOUNDRY_SERVICE_ERROR_ALREADY_STOPPED,
                                  "Service already stopped");

  priv->has_stopped = TRUE;

  g_debug ("Stopping service %s", G_OBJECT_TYPE_NAME (self));

  future = FOUNDRY_SERVICE_GET_CLASS (self)->stop (self);
  future = dex_future_finally (future,
                               foundry_service_propagate,
                               dex_ref (priv->stopped),
                               dex_unref);

  return future;
}

void
foundry_service_class_set_action_prefix (FoundryServiceClass *service_class,
                                         const char          *action_prefix)
{
  FoundryServiceClassPrivate *priv;

  g_return_if_fail (FOUNDRY_IS_SERVICE_CLASS (service_class));

  priv = foundry_service_class_get_private (service_class);
  priv->action_prefix = g_intern_string (action_prefix);
}

const char *
foundry_service_class_get_action_prefix (FoundryServiceClass *service_class)
{
  FoundryServiceClassPrivate *priv;

  g_return_val_if_fail (FOUNDRY_IS_SERVICE_CLASS (service_class), NULL);

  priv = foundry_service_class_get_private (service_class);

  return priv->action_prefix;
}

GActionGroup *
foundry_service_get_action_group (FoundryService *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SERVICE (self), NULL);

  return G_ACTION_GROUP (foundry_action_mixin_get_action_muxer (self));
}

/**
 * foundry_service_class_install_action:
 * @parameter_type: (nullable):
 * @activate: (scope forever):
 */
void
foundry_service_class_install_action (FoundryServiceClass  *service_class,
                                      const char           *action_name,
                                      const char           *parameter_type,
                                      FoundryServiceAction  activate)
{
  FoundryServiceClassPrivate *priv;

  g_return_if_fail (FOUNDRY_IS_SERVICE_CLASS (service_class));
  g_return_if_fail (action_name != NULL);
  g_return_if_fail (activate != NULL);

  priv = foundry_service_class_get_private (service_class);

  foundry_action_mixin_install_action (&priv->actions, action_name, parameter_type,
                                       (FoundryActionActivateFunc)activate);
}

gboolean
foundry_service_action_get_enabled (FoundryService *self,
                                    const char     *action_name)
{
  g_return_val_if_fail (FOUNDRY_IS_SERVICE (self), FALSE);
  g_return_val_if_fail (action_name != NULL, FALSE);

  return foundry_action_mixin_get_enabled (self, action_name);
}

void
foundry_service_action_set_enabled (FoundryService *self,
                                    const char     *action_name,
                                    gboolean        enabled)
{
  g_return_if_fail (FOUNDRY_IS_SERVICE (self));
  g_return_if_fail (action_name != NULL);

  foundry_action_mixin_set_enabled (self, action_name, enabled);
}
