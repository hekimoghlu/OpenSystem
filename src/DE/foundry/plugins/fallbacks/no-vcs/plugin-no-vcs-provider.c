/* plugin-no-vcs-provider.c
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

#include "plugin-no-vcs-provider.h"

struct _PluginNoVcsProvider
{
  FoundryVcsProvider  parent_instance;
  FoundryVcs         *vcs;
};

G_DEFINE_FINAL_TYPE (PluginNoVcsProvider, plugin_no_vcs_provider, FOUNDRY_TYPE_VCS_PROVIDER)

static DexFuture *
plugin_no_vcs_provider_load (FoundryVcsProvider *provider)
{
  PluginNoVcsProvider *self = (PluginNoVcsProvider *)provider;
  g_autoptr(FoundryContext) context = NULL;

  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_NO_VCS_PROVIDER (self));

  if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    {
      self->vcs = foundry_no_vcs_new (context);
      foundry_vcs_provider_set_vcs (FOUNDRY_VCS_PROVIDER (self), self->vcs);
    }

  FOUNDRY_RETURN (dex_future_new_true ());
}

static DexFuture *
plugin_no_vcs_provider_unload (FoundryVcsProvider *provider)
{
  PluginNoVcsProvider *self = (PluginNoVcsProvider *)provider;
  g_autoptr(FoundryContext) context = NULL;

  FOUNDRY_ENTRY;

  g_assert (FOUNDRY_IS_MAIN_THREAD ());
  g_assert (PLUGIN_IS_NO_VCS_PROVIDER (self));

  if (self->vcs != NULL)
    {
      foundry_vcs_provider_set_vcs (FOUNDRY_VCS_PROVIDER (self), NULL);
      g_clear_object (&self->vcs);
    }

  FOUNDRY_RETURN (dex_future_new_true ());
}

static void
plugin_no_vcs_provider_finalize (GObject *object)
{
  PluginNoVcsProvider *self = (PluginNoVcsProvider *)object;

  g_clear_object (&self->vcs);

  G_OBJECT_CLASS (plugin_no_vcs_provider_parent_class)->finalize (object);
}

static void
plugin_no_vcs_provider_class_init (PluginNoVcsProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryVcsProviderClass *vcs_provider_class = FOUNDRY_VCS_PROVIDER_CLASS (klass);

  object_class->finalize = plugin_no_vcs_provider_finalize;

  vcs_provider_class->load = plugin_no_vcs_provider_load;
  vcs_provider_class->unload = plugin_no_vcs_provider_unload;
}

static void
plugin_no_vcs_provider_init (PluginNoVcsProvider *self)
{
}
