/* plugin-flatpak-sdk-provider.c
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

#include "plugin-flatpak.h"
#include "plugin-flatpak-sdk.h"
#include "plugin-flatpak-sdk-provider.h"
#include "plugin-flatpak-util.h"

struct _PluginFlatpakSdkProvider
{
  FoundrySdkProvider  parent_instance;
  GPtrArray          *installations;
};

G_DEFINE_FINAL_TYPE (PluginFlatpakSdkProvider, plugin_flatpak_sdk_provider, FOUNDRY_TYPE_SDK_PROVIDER)

static DexFuture *
plugin_flatpak_sdk_provider_load_fiber (gpointer user_data)
{
  FlatpakQueryFlags flags = 0;
  PluginFlatpakSdkProvider *self = user_data;
  FlatpakInstallation *installation;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) installations = NULL;
  g_autoptr(GPtrArray) futures_installations = NULL;
  g_autoptr(GPtrArray) futures = NULL;

  g_assert (PLUGIN_IS_FLATPAK_SDK_PROVIDER (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));

  if ((installations = dex_await_boxed (plugin_flatpak_load_installations (), NULL)))
    {
      g_autoptr(FlatpakInstallation) private_installation = NULL;

      if (self->installations->len > 0)
        g_ptr_array_remove_range (self->installations, 0, self->installations->len);

      for (guint i = 0; i < installations->len; i++)
        g_ptr_array_add (self->installations, g_object_ref (g_ptr_array_index (installations, i)));

      /* Create a private instance for this context which may have overrided
       * the private installation location.
       */
      if ((private_installation = dex_await_object (plugin_flatpak_installation_new_private (context), NULL)))
        g_ptr_array_add (self->installations, g_steal_pointer (&private_installation));
    }

  futures = g_ptr_array_new_with_free_func (dex_unref);
  futures_installations = g_ptr_array_new_with_free_func (g_object_unref);
  flags = FLATPAK_QUERY_FLAGS_ONLY_CACHED | FLATPAK_QUERY_FLAGS_ALL_ARCHES;

  /* During load we only show installed refs. We will queue
   * the other refs to get loaded later. In most cases we
   * only need the installed ones anyway.
   */
  for (guint i = 0; i < self->installations->len; i++)
    {
      installation = g_ptr_array_index (self->installations, i);

      g_ptr_array_add (futures_installations, g_object_ref (installation));
      g_ptr_array_add (futures, plugin_flatpak_installation_list_installed_refs (context, installation, flags));
    }

  if (futures->len > 0)
    dex_await (foundry_future_all (futures), NULL);

  for (guint i = 0; i < futures->len; i++)
    {
      DexFuture *future = g_ptr_array_index (futures, i);
      g_autoptr(GPtrArray) refs = NULL;

      installation = g_ptr_array_index (futures_installations, i);

      if (!(refs = dex_await_boxed (dex_ref (future), NULL)))
        continue;

      for (guint j = 0; j < refs->len; j++)
        {
          FlatpakRef *ref;

          ref = g_ptr_array_index (refs, j);

          if (plugin_flatpak_ref_can_be_sdk (ref))
            {
              g_autoptr(PluginFlatpakSdk) sdk = plugin_flatpak_sdk_new (context, installation, ref);

              foundry_sdk_provider_sdk_added (FOUNDRY_SDK_PROVIDER (self), FOUNDRY_SDK (sdk));
            }
        }
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_flatpak_sdk_provider_load (FoundrySdkProvider *sdk_provider)
{
  PluginFlatpakSdkProvider *self = (PluginFlatpakSdkProvider *)sdk_provider;

  g_assert (PLUGIN_IS_FLATPAK_SDK_PROVIDER (self));

  return dex_scheduler_spawn (NULL, 0,
                              plugin_flatpak_sdk_provider_load_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

static DexFuture *
plugin_flatpak_sdk_provider_unload (FoundrySdkProvider *sdk_provider)
{
  PluginFlatpakSdkProvider *self = (PluginFlatpakSdkProvider *)sdk_provider;

  g_assert (PLUGIN_IS_FLATPAK_SDK_PROVIDER (self));

  return FOUNDRY_SDK_PROVIDER_CLASS (plugin_flatpak_sdk_provider_parent_class)->unload (sdk_provider);
}

typedef struct _FindById
{
  PluginFlatpakSdkProvider *self;
  char *sdk_id;
  char *sdk_name;
  char *sdk_arch;
  char *sdk_branch;
} FindById;

static void
find_by_id_free (FindById *state)
{
  g_clear_object (&state->self);
  g_clear_pointer (&state->sdk_id, g_free);
  g_clear_pointer (&state->sdk_name, g_free);
  g_clear_pointer (&state->sdk_arch, g_free);
  g_clear_pointer (&state->sdk_branch, g_free);
  g_free (state);
}

static DexFuture *
plugin_flatpak_sdk_provider_find_by_id_fiber (gpointer data)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) installations = NULL;
  g_autofree char *arch = NULL;
  FlatpakQueryFlags flags = 0;
  FindById *state = data;

  g_assert (state != NULL);
  g_assert (PLUGIN_IS_FLATPAK_SDK_PROVIDER (state->self));
  g_assert (state->sdk_id != NULL);
  g_assert (state->sdk_name != NULL);
  g_assert (state->sdk_arch != NULL);
  g_assert (state->sdk_branch != NULL);

  arch = g_strdup_printf ("/%s/", flatpak_get_default_arch ());

  if (strstr (state->sdk_id, arch) == NULL)
    flags |= FLATPAK_QUERY_FLAGS_ALL_ARCHES;

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (state->self));
  installations = g_ptr_array_ref (state->self->installations);

  for (guint i = 0; i < installations->len; i++)
    {
      g_autoptr(FlatpakInstallation) installation = g_object_ref (g_ptr_array_index (installations, i));
      g_autoptr(GPtrArray) refs = NULL;

      if ((refs = dex_await_boxed (plugin_flatpak_installation_list_refs  (context, installation, flags), NULL)))
        {
          for (guint j = 0; j < refs->len; j++)
            {
              FlatpakRef *ref = g_ptr_array_index (refs, j);

              if (plugin_flatpak_ref_matches (ref, state->sdk_name, state->sdk_arch, state->sdk_branch))
                return dex_future_new_take_object (plugin_flatpak_sdk_new (context, installation, ref));
            }
        }
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

static DexFuture *
plugin_flatpak_sdk_provider_find_by_id (FoundrySdkProvider *sdk_provider,
                                        const char         *sdk_id)
{
  g_autofree char *name = NULL;
  g_autofree char *arch = NULL;
  g_autofree char *branch = NULL;
  FindById *state;

  g_assert (PLUGIN_IS_FLATPAK_SDK_PROVIDER (sdk_provider));
  g_assert (sdk_id != NULL);

  if (!plugin_flatpak_split_id (sdk_id, &name, &arch, &branch) ||
      name == NULL || arch == NULL || branch == NULL)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "Not found");

  state = g_new0 (FindById, 1);
  state->self = g_object_ref (PLUGIN_FLATPAK_SDK_PROVIDER (sdk_provider));
  state->sdk_id = g_strdup (sdk_id);
  state->sdk_name = g_steal_pointer (&name);
  state->sdk_arch = g_steal_pointer (&arch);
  state->sdk_branch = g_steal_pointer (&branch);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_flatpak_sdk_provider_find_by_id_fiber,
                              state,
                              (GDestroyNotify)find_by_id_free);
}

static void
plugin_flatpak_sdk_provider_finalize (GObject *object)
{
  PluginFlatpakSdkProvider *self = (PluginFlatpakSdkProvider *)object;

  g_clear_pointer (&self->installations, g_ptr_array_unref);

  G_OBJECT_CLASS (plugin_flatpak_sdk_provider_parent_class)->finalize (object);
}

static void
plugin_flatpak_sdk_provider_class_init (PluginFlatpakSdkProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundrySdkProviderClass *sdk_provider_class = FOUNDRY_SDK_PROVIDER_CLASS (klass);

  object_class->finalize = plugin_flatpak_sdk_provider_finalize;

  sdk_provider_class->load = plugin_flatpak_sdk_provider_load;
  sdk_provider_class->unload = plugin_flatpak_sdk_provider_unload;
  sdk_provider_class->find_by_id = plugin_flatpak_sdk_provider_find_by_id;
}

static void
plugin_flatpak_sdk_provider_init (PluginFlatpakSdkProvider *self)
{
  self->installations = g_ptr_array_new_with_free_func (g_object_unref);
}
