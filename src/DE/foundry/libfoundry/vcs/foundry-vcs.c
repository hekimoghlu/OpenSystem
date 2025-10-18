/* foundry-vcs.c
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

#include "foundry-vcs-blame.h"
#include "foundry-vcs-file.h"
#include "foundry-vcs-manager.h"
#include "foundry-vcs-private.h"
#include "foundry-vcs-remote.h"
#include "foundry-vcs-tree.h"
#include "foundry-util.h"

typedef struct _FoundryVcsPrivate
{
  GWeakRef provider_wr;
} FoundryVcsPrivate;

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryVcs, foundry_vcs, FOUNDRY_TYPE_CONTEXTUAL)

enum {
  PROP_0,
  PROP_ACTIVE,
  PROP_BRANCH_NAME,
  PROP_ID,
  PROP_NAME,
  PROP_PRIORITY,
  PROP_PROVIDER,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static DexFuture *
foundry_vcs_real_find_remote_cb (DexFuture *future,
                                 gpointer   user_data)
{
  const char *name = user_data;
  g_autoptr(GListModel) model = NULL;
  guint n_items;

  g_assert (DEX_IS_FUTURE (future));
  g_assert (name != NULL);

  model = dex_await_object (dex_ref (future), NULL);

  g_assert (model != NULL);
  g_assert (G_IS_LIST_MODEL (model));

  n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryVcsRemote) remote = g_list_model_get_item (model, i);
      g_autofree char *remote_name = foundry_vcs_remote_dup_name (remote);

      if (g_strcmp0 (remote_name, name) == 0)
        return dex_future_new_take_object (g_steal_pointer (&remote));
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

static DexFuture *
foundry_vcs_real_find_remote (FoundryVcs *self,
                              const char *name)
{
  return dex_future_then (foundry_vcs_list_remotes (self),
                          foundry_vcs_real_find_remote_cb,
                          g_strdup (name),
                          g_free);
}

static gboolean
foundry_vcs_real_is_file_ignored (FoundryVcs *self,
                                  GFile      *file)
{
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GFile) project_dir = NULL;

  g_assert (FOUNDRY_IS_VCS (self));
  g_assert (G_IS_FILE (file));

  if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))) &&
      (project_dir = foundry_context_dup_project_directory (context)) &&
      g_file_has_prefix (file, project_dir))
    {
      g_autofree char *relative_path = g_file_get_relative_path (project_dir, file);
      return foundry_vcs_is_ignored (self, relative_path);
    }

  return FALSE;
}

static void
foundry_vcs_finalize (GObject *object)
{
  FoundryVcs *self = (FoundryVcs *)object;
  FoundryVcsPrivate *priv = foundry_vcs_get_instance_private (self);

  g_weak_ref_clear (&priv->provider_wr);

  G_OBJECT_CLASS (foundry_vcs_parent_class)->finalize (object);
}

static void
foundry_vcs_get_property (GObject    *object,
                          guint       prop_id,
                          GValue     *value,
                          GParamSpec *pspec)
{
  FoundryVcs *self = FOUNDRY_VCS (object);

  switch (prop_id)
    {
    case PROP_ACTIVE:
      g_value_set_boolean (value, foundry_vcs_get_active (self));
      break;

    case PROP_BRANCH_NAME:
      g_value_take_string (value, foundry_vcs_dup_branch_name (self));
      break;

    case PROP_ID:
      g_value_take_string (value, foundry_vcs_dup_id (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_vcs_dup_name (self));
      break;

    case PROP_PRIORITY:
      g_value_set_uint (value, foundry_vcs_get_priority (self));
      break;

    case PROP_PROVIDER:
      g_value_take_object (value, _foundry_vcs_dup_provider (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_vcs_class_init (FoundryVcsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_vcs_finalize;
  object_class->get_property = foundry_vcs_get_property;

  klass->is_file_ignored = foundry_vcs_real_is_file_ignored;
  klass->find_remote = foundry_vcs_real_find_remote;

  properties[PROP_ACTIVE] =
    g_param_spec_boolean ("active", NULL, NULL,
                         FALSE,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_BRANCH_NAME] =
    g_param_spec_string ("branch-name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PRIORITY] =
    g_param_spec_uint ("priority", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_PROVIDER] =
    g_param_spec_object ("provider", NULL, NULL,
                         FOUNDRY_TYPE_VCS_PROVIDER,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_vcs_init (FoundryVcs *self)
{
}

gboolean
foundry_vcs_get_active (FoundryVcs *self)
{
  g_autoptr(FoundryContext) context = NULL;

  g_return_val_if_fail (FOUNDRY_IS_VCS (self), FALSE);

  if ((context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self))))
    {
      g_autoptr(FoundryVcsManager) vcs_manager = foundry_context_dup_vcs_manager (context);
      g_autoptr(FoundryVcs) vcs = foundry_vcs_manager_dup_vcs (vcs_manager);

      return vcs == self;
    }

  return FALSE;
}

/**
 * foundry_vcs_dup_id:
 * @self: a #FoundryVcs
 *
 * Gets the identifier for the VCS such as "git" or "none".
 *
 * Returns: (transfer full): a string containing the identifier
 */
char *
foundry_vcs_dup_id (FoundryVcs *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS (self), NULL);

  return FOUNDRY_VCS_GET_CLASS (self)->dup_id (self);
}

/**
 * foundry_vcs_dup_name:
 * @self: a #FoundryVcs
 *
 * Gets the name of the vcs in title format such as "Git"
 *
 * Returns: (transfer full): a string containing the name
 */
char *
foundry_vcs_dup_name (FoundryVcs *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS (self), NULL);

  return FOUNDRY_VCS_GET_CLASS (self)->dup_name (self);
}

/**
 * foundry_vcs_dup_branch_name:
 * @self: a #FoundryVcs
 *
 * Gets the name of the branch such as "main".
 *
 * Returns: (transfer full): a string containing the branch name
 */
char *
foundry_vcs_dup_branch_name (FoundryVcs *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS (self), NULL);

  return FOUNDRY_VCS_GET_CLASS (self)->dup_branch_name (self);
}

FoundryVcsProvider *
_foundry_vcs_dup_provider (FoundryVcs *self)
{
  FoundryVcsPrivate *priv = foundry_vcs_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_VCS (self), NULL);

  return g_weak_ref_get (&priv->provider_wr);
}

void
_foundry_vcs_set_provider (FoundryVcs         *self,
                           FoundryVcsProvider *provider)
{
  FoundryVcsPrivate *priv = foundry_vcs_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_VCS (self));
  g_return_if_fail (!provider || FOUNDRY_IS_VCS_PROVIDER (provider));

  g_weak_ref_set (&priv->provider_wr, provider);
}

guint
foundry_vcs_get_priority (FoundryVcs *self)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS (self), 0);

  if (FOUNDRY_VCS_GET_CLASS (self)->get_priority)
    return FOUNDRY_VCS_GET_CLASS (self)->get_priority (self);

  return 0;
}

gboolean
foundry_vcs_is_ignored (FoundryVcs *self,
                        const char *relative_path)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS (self), FALSE);
  g_return_val_if_fail (relative_path != NULL, FALSE);

  if (FOUNDRY_VCS_GET_CLASS (self)->is_ignored)
    return FOUNDRY_VCS_GET_CLASS (self)->is_ignored (self, relative_path);

  return FALSE;
}

gboolean
foundry_vcs_is_file_ignored (FoundryVcs *self,
                             GFile      *file)
{
  g_return_val_if_fail (FOUNDRY_IS_VCS (self), FALSE);
  g_return_val_if_fail (G_IS_FILE (file), FALSE);

  return FOUNDRY_VCS_GET_CLASS (self)->is_file_ignored (self, file);
}

/**
 * foundry_vcs_list_files:
 * @self: a [class@Foundry.Vcs]
 *
 * List all files in the repository.
 *
 * It is not required that implementations return files that are not
 * indexed in their caches from this method.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves
 *   to [iface@Gio.ListModel] of [class@Foundry.VcsFile].
 */
DexFuture *
foundry_vcs_list_files (FoundryVcs *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));

  if (FOUNDRY_VCS_GET_CLASS (self)->list_files)
    return FOUNDRY_VCS_GET_CLASS (self)->list_files (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_find_file:
 * @self: a [class@Foundry.Vcs]
 * @file: the file to locate
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.VcsFile] or rejects with error
 */
DexFuture *
foundry_vcs_find_file (FoundryVcs *self,
                       GFile      *file)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));

  if (FOUNDRY_VCS_GET_CLASS (self)->find_file)
    return FOUNDRY_VCS_GET_CLASS (self)->find_file (self, file);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_blame:
 * @self: a [class@Foundry.Vcs]
 * @file: a [class@Foundry.VcsFile]
 * @bytes: (nullable): optional contents for the file
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.VcsBlame] or rejects with error
 */
DexFuture *
foundry_vcs_blame (FoundryVcs     *self,
                   FoundryVcsFile *file,
                   GBytes         *bytes)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));

  if (FOUNDRY_VCS_GET_CLASS (self)->blame)
    return FOUNDRY_VCS_GET_CLASS (self)->blame (self, file, bytes);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_list_branches:
 * @self: a [class@Foundry.Vcs]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.VcsBranch].
 */
DexFuture *
foundry_vcs_list_branches (FoundryVcs *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));

  if (FOUNDRY_VCS_GET_CLASS (self)->list_branches)
    return FOUNDRY_VCS_GET_CLASS (self)->list_branches (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_list_tags:
 * @self: a [class@Foundry.Vcs]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.VcsTag].
 */
DexFuture *
foundry_vcs_list_tags (FoundryVcs *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));

  if (FOUNDRY_VCS_GET_CLASS (self)->list_tags)
    return FOUNDRY_VCS_GET_CLASS (self)->list_tags (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_list_remotes:
 * @self: a [class@Foundry.Vcs]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.VcsRemote].
 */
DexFuture *
foundry_vcs_list_remotes (FoundryVcs *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));

  if (FOUNDRY_VCS_GET_CLASS (self)->list_remotes)
    return FOUNDRY_VCS_GET_CLASS (self)->list_remotes (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_fetch:
 * @self: a [class@Foundry.Vcs]
 * @remote: a [class@Foundry.VcsRemote]
 * @operation: a [class@Foundry.Operation]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value or rejects with error.
 */
DexFuture *
foundry_vcs_fetch (FoundryVcs       *self,
                   FoundryVcsRemote *remote,
                   FoundryOperation *operation)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));
  dex_return_error_if_fail (FOUNDRY_IS_VCS_REMOTE (remote));
  dex_return_error_if_fail (FOUNDRY_IS_OPERATION (operation));

  if (FOUNDRY_VCS_GET_CLASS (self)->fetch)
    return FOUNDRY_VCS_GET_CLASS (self)->fetch (self, remote, operation);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_find_remote:
 * @self: a [class@Foundry.Vcs]
 * @name: the name of the remote
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.VcsRemote].
 */
DexFuture *
foundry_vcs_find_remote (FoundryVcs *self,
                         const char *name)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));
  dex_return_error_if_fail (name != NULL);

  if (FOUNDRY_VCS_GET_CLASS (self)->find_remote)
    return FOUNDRY_VCS_GET_CLASS (self)->find_remote (self, name);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_find_commit:
 * @self: a [class@Foundry.Vcs]
 * @id: the identifier of the commit
 *
 * Finds a commit by identifier
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.VcsCommit].
 */
DexFuture *
foundry_vcs_find_commit (FoundryVcs *self,
                         const char *id)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));
  dex_return_error_if_fail (id != NULL);

  if (FOUNDRY_VCS_GET_CLASS (self)->find_commit)
    return FOUNDRY_VCS_GET_CLASS (self)->find_commit (self, id);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_find_tree:
 * @self: a [class@Foundry.Vcs]
 * @id: the identifier of the tree
 *
 * Finds a [class@Foundry.VcsTree] by tree identifier
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.VcsTree].
 */
DexFuture *
foundry_vcs_find_tree (FoundryVcs *self,
                       const char *id)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));
  dex_return_error_if_fail (id != NULL);

  if (FOUNDRY_VCS_GET_CLASS (self)->find_tree)
    return FOUNDRY_VCS_GET_CLASS (self)->find_tree (self, id);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_list_commits_with_file:
 * @self: a [class@Foundry.Vcs]
 * @file: a [class@Foundry.VcsFile]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [iface@Gio.ListModel] of [class@Foundry.VcsCommit]
 */
DexFuture *
foundry_vcs_list_commits_with_file (FoundryVcs     *self,
                                    FoundryVcsFile *file)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));
  dex_return_error_if_fail (FOUNDRY_IS_VCS_FILE (file));

  if (FOUNDRY_VCS_GET_CLASS (self)->list_commits_with_file)
    return FOUNDRY_VCS_GET_CLASS (self)->list_commits_with_file (self, file);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_diff:
 * @self: a [class@Foundry.Vcs]
 * @tree_a: the old tree
 * @tree_b: the new tree
 *
 * Diffs two [class@Foundry.VcsTree] resulting in a [class@Foundry.VcsDiff]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.VcsDiff] or rejects with error.
 */
DexFuture *
foundry_vcs_diff (FoundryVcs     *self,
                  FoundryVcsTree *tree_a,
                  FoundryVcsTree *tree_b)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));
  dex_return_error_if_fail (FOUNDRY_IS_VCS_TREE (tree_a));
  dex_return_error_if_fail (FOUNDRY_IS_VCS_TREE (tree_b));

  if (FOUNDRY_VCS_GET_CLASS (self)->diff)
    return FOUNDRY_VCS_GET_CLASS (self)->diff (self, tree_a, tree_b);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_describe_line_changes:
 * @self: a [class@Foundry.Vcs]
 * @file: a [class@Foundry.VcsFile]
 * @contents: the contents of the file
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [class@Foundry.VcsLineChanges] or rejects with error.
 */
DexFuture *
foundry_vcs_describe_line_changes (FoundryVcs     *self,
                                   FoundryVcsFile *file,
                                   GBytes         *contents)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));
  dex_return_error_if_fail (FOUNDRY_IS_VCS_FILE (file));
  dex_return_error_if_fail (contents != NULL);

  if (FOUNDRY_VCS_GET_CLASS (self)->describe_line_changes)
    return FOUNDRY_VCS_GET_CLASS (self)->describe_line_changes (self, file, contents);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_vcs_query_file_status:
 *
 * Queries the state of @file in the repository.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [flags@Foundry.VcsFileStatus] or rejects with error.
 */
DexFuture *
foundry_vcs_query_file_status (FoundryVcs *self,
                               GFile      *file)
{
  dex_return_error_if_fail (FOUNDRY_IS_VCS (self));
  dex_return_error_if_fail (G_IS_FILE (file));

  if (FOUNDRY_VCS_GET_CLASS (self)->query_file_status)
    return FOUNDRY_VCS_GET_CLASS (self)->query_file_status (self, file);

  return foundry_future_new_not_supported ();
}

G_DEFINE_FLAGS_TYPE (FoundryVcsFileStatus, foundry_vcs_file_status,
                     G_DEFINE_ENUM_VALUE (FOUNDRY_VCS_FILE_STATUS_CURRENT, "current"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_VCS_FILE_STATUS_MODIFIED_IN_STAGE, "modified-in-stage"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_VCS_FILE_STATUS_MODIFIED_IN_TREE, "modified-in-tree"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_VCS_FILE_STATUS_NEW_IN_STAGE, "new-in-stage"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_VCS_FILE_STATUS_NEW_IN_TREE, "new-in-tree"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_VCS_FILE_STATUS_DELETED_IN_STAGE, "deleted-in-stage"),
                     G_DEFINE_ENUM_VALUE (FOUNDRY_VCS_FILE_STATUS_DELETED_IN_TREE, "deleted-in-tree"))
