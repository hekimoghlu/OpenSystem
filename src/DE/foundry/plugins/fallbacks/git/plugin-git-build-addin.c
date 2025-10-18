/* plugin-git-build-addin.c
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

#include "plugin-git-build-addin.h"

struct _PluginGitBuildAddin
{
  FoundryBuildAddin parent_instance;
};

G_DEFINE_FINAL_TYPE (PluginGitBuildAddin, plugin_git_build_addin, FOUNDRY_TYPE_BUILD_ADDIN)

static DexFuture *
plugin_git_build_addin_load (FoundryBuildAddin *build_addin)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;

  g_assert (PLUGIN_IS_GIT_BUILD_ADDIN (build_addin));

  pipeline = foundry_build_addin_dup_pipeline (build_addin);

  /* TODO: Create submodule init/update stage */

  return dex_future_new_true ();
}

static DexFuture *
plugin_git_build_addin_unload (FoundryBuildAddin *build_addin)
{
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;

  g_assert (PLUGIN_IS_GIT_BUILD_ADDIN (build_addin));

  pipeline = foundry_build_addin_dup_pipeline (build_addin);

  /* TODO: Remove submodule init/update stage */

  return FOUNDRY_BUILD_ADDIN_CLASS (plugin_git_build_addin_parent_class)->unload (build_addin);
}

static void
plugin_git_build_addin_class_init (PluginGitBuildAddinClass *klass)
{
  FoundryBuildAddinClass *build_addin_class = FOUNDRY_BUILD_ADDIN_CLASS (klass);

  build_addin_class->load = plugin_git_build_addin_load;
  build_addin_class->unload = plugin_git_build_addin_unload;
}

static void
plugin_git_build_addin_init (PluginGitBuildAddin *self)
{
}
