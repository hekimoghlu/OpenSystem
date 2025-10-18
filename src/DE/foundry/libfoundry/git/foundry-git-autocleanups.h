/* foundry-git-autocleanups.h
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

#pragma once

#include <glib.h>
#include <git2.h>

G_BEGIN_DECLS

G_DEFINE_AUTO_CLEANUP_CLEAR_FUNC (git_buf, git_buf_dispose)
G_DEFINE_AUTO_CLEANUP_CLEAR_FUNC (git_strarray, git_strarray_dispose)

G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_blame, git_blame_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_blob, git_blob_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_branch_iterator, git_branch_iterator_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_commit, git_commit_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_config, git_config_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_config_entry, git_config_entry_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_diff, git_diff_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_diff_stats, git_diff_stats_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_index, git_index_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_object, git_object_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_reference, git_reference_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_reference_iterator, git_reference_iterator_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_remote, git_remote_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_repository, git_repository_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_revwalk, git_revwalk_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_signature, git_signature_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_status_list, git_status_list_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_tree, git_tree_free)
G_DEFINE_AUTOPTR_CLEANUP_FUNC (git_tree_entry, git_tree_entry_free)

G_END_DECLS
