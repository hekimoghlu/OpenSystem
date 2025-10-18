/* plugin-git-provider.c
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

#include <git2.h>

#include "foundry-git-autocleanups.h"
#include "foundry-git-error.h"
#include "foundry-git-vcs-private.h"

#include "plugin-git-vcs-provider.h"

struct _PluginGitVcsProvider
{
  FoundryVcsProvider  parent_instance;
  FoundryGitVcs      *vcs;
};

G_DEFINE_FINAL_TYPE (PluginGitVcsProvider, plugin_git_vcs_provider, FOUNDRY_TYPE_VCS_PROVIDER)

static DexFuture *
plugin_git_vcs_provider_load_fiber (gpointer data)
{
  PluginGitVcsProvider *self = data;
  g_autoptr(FoundryVcsManager) vcs_manager = NULL;
  g_autoptr(git_repository) repository = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryGitVcs) vcs = NULL;
  g_autoptr(GError) error = NULL;
  g_autoptr(GFile) project_dir = NULL;
  g_auto(git_buf) buf = GIT_BUF_INIT;
  const char *path;

  g_assert (PLUGIN_IS_GIT_VCS_PROVIDER (self));

  if (!(context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    return dex_future_new_true ();

  project_dir = foundry_context_dup_project_directory (context);

  if (!g_file_is_native (project_dir))
    return dex_future_new_true ();

  path = g_file_peek_path (project_dir);

  foundry_git_return_if_error (git_repository_discover (&buf, path, TRUE, NULL));
  foundry_git_return_if_error (git_repository_open (&repository, buf.ptr));

  if (!(vcs = dex_await_object (_foundry_git_vcs_new (context, g_steal_pointer (&repository)), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  foundry_vcs_provider_set_vcs (FOUNDRY_VCS_PROVIDER (self), FOUNDRY_VCS (vcs));

  return dex_future_new_true ();
}

static DexFuture *
plugin_git_vcs_provider_load (FoundryVcsProvider *provider)
{
  g_assert (PLUGIN_IS_GIT_VCS_PROVIDER (provider));

  return dex_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                              plugin_git_vcs_provider_load_fiber,
                              g_object_ref (provider),
                              g_object_unref);
}

static DexFuture *
plugin_git_vcs_provider_unload (FoundryVcsProvider *provider)
{
  PluginGitVcsProvider *self = (PluginGitVcsProvider *)provider;

  g_assert (PLUGIN_IS_GIT_VCS_PROVIDER (self));

  foundry_vcs_provider_set_vcs (FOUNDRY_VCS_PROVIDER (self), NULL);

  g_clear_object (&self->vcs);

  return dex_future_new_true ();
}

static void
plugin_git_vcs_provider_class_init (PluginGitVcsProviderClass *klass)
{
  FoundryVcsProviderClass *vcs_provider_class = FOUNDRY_VCS_PROVIDER_CLASS (klass);

  vcs_provider_class->load = plugin_git_vcs_provider_load;
  vcs_provider_class->unload = plugin_git_vcs_provider_unload;
}

static void
plugin_git_vcs_provider_init (PluginGitVcsProvider *self)
{
}
