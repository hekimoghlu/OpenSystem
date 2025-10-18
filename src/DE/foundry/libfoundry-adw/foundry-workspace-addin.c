/* foundry-workspace-addin.c
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

#include "foundry-workspace-addin-private.h"

typedef struct
{
  GWeakRef workspace_wr;
} FoundryWorkspaceAddinPrivate;

enum {
  PROP_0,
  PROP_WORKSPACE,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryWorkspaceAddin, foundry_workspace_addin, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static DexFuture *
foundry_workspace_addin_real_load (FoundryWorkspaceAddin *self)
{
  return dex_future_new_true ();
}

static DexFuture *
foundry_workspace_addin_real_unload (FoundryWorkspaceAddin *self)
{
  return dex_future_new_true ();
}

static void
foundry_workspace_addin_dispose (GObject *object)
{
  FoundryWorkspaceAddin *self = (FoundryWorkspaceAddin *)object;
  FoundryWorkspaceAddinPrivate *priv = foundry_workspace_addin_get_instance_private (self);

  g_weak_ref_set (&priv->workspace_wr, NULL);

  G_OBJECT_CLASS (foundry_workspace_addin_parent_class)->dispose (object);
}

static void
foundry_workspace_addin_finalize (GObject *object)
{
  FoundryWorkspaceAddin *self = (FoundryWorkspaceAddin *)object;
  FoundryWorkspaceAddinPrivate *priv = foundry_workspace_addin_get_instance_private (self);

  g_weak_ref_clear (&priv->workspace_wr);

  G_OBJECT_CLASS (foundry_workspace_addin_parent_class)->finalize (object);
}

static void
foundry_workspace_addin_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  FoundryWorkspaceAddin *self = FOUNDRY_WORKSPACE_ADDIN (object);

  switch (prop_id)
    {
    case PROP_WORKSPACE:
      g_value_take_object (value, foundry_workspace_addin_dup_workspace (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_workspace_addin_class_init (FoundryWorkspaceAddinClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_workspace_addin_dispose;
  object_class->finalize = foundry_workspace_addin_finalize;
  object_class->get_property = foundry_workspace_addin_get_property;

  klass->load = foundry_workspace_addin_real_load;
  klass->unload = foundry_workspace_addin_real_unload;

  properties[PROP_WORKSPACE] =
    g_param_spec_object ("workspace", NULL, NULL,
                         FOUNDRY_TYPE_WORKSPACE,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_workspace_addin_init (FoundryWorkspaceAddin *self)
{
}

/**
 * foundry_workspace_addin_dup_workspace:
 * @self: a [class@FoundryAdw.WorkspaceAddin]
 *
 * Returns: (transfer full):
 */
FoundryWorkspace *
foundry_workspace_addin_dup_workspace (FoundryWorkspaceAddin *self)
{
  FoundryWorkspaceAddinPrivate *priv = foundry_workspace_addin_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE_ADDIN (self), NULL);

  return g_weak_ref_get (&priv->workspace_wr);
}

DexFuture *
_foundry_workspace_addin_load (FoundryWorkspaceAddin *self,
                               FoundryWorkspace      *workspace)
{
  FoundryWorkspaceAddinPrivate *priv = foundry_workspace_addin_get_instance_private (self);

  dex_return_error_if_fail (FOUNDRY_IS_WORKSPACE_ADDIN (self));
  dex_return_error_if_fail (FOUNDRY_IS_WORKSPACE (workspace));

  g_weak_ref_set (&priv->workspace_wr, workspace);
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_WORKSPACE]);

  return FOUNDRY_WORKSPACE_ADDIN_GET_CLASS (self)->load (self);
}

static DexFuture *
foundry_workspace_addin_clear_workspace (DexFuture *future,
                                         gpointer   user_data)
{
  FoundryWorkspaceAddin *self = user_data;
  FoundryWorkspaceAddinPrivate *priv = foundry_workspace_addin_get_instance_private (self);

  g_assert (FOUNDRY_IS_WORKSPACE_ADDIN (self));

  g_weak_ref_set (&priv->workspace_wr, NULL);
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_WORKSPACE]);

  return dex_ref (future);
}

DexFuture *
_foundry_workspace_addin_unload (FoundryWorkspaceAddin *self)
{
  DexFuture *future;

  dex_return_error_if_fail (FOUNDRY_IS_WORKSPACE_ADDIN (self));

  future = FOUNDRY_WORKSPACE_ADDIN_GET_CLASS (self)->unload (self);
  future = dex_future_finally (dex_ref (future),
                               foundry_workspace_addin_clear_workspace,
                               g_object_ref (self),
                               g_object_unref);

  dex_future_disown (dex_ref (future));

  return g_steal_pointer (&future);
}
