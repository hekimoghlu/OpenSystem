/* foundry-git-error.h
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

#include <git2.h>
#include <libdex.h>

G_BEGIN_DECLS

#define FOUNDRY_GIT_ERROR (foundry_git_error_quark())

GQuark     foundry_git_error_quark       (void) G_GNUC_CONST;
DexFuture *foundry_git_reject_last_error (void) G_GNUC_WARN_UNUSED_RESULT;

#define foundry_git_return_if_error(check) \
  G_STMT_START { \
    int __val = (check); \
    if (__val != GIT_OK) \
      return foundry_git_reject_last_error (); \
  } G_STMT_END

G_END_DECLS
