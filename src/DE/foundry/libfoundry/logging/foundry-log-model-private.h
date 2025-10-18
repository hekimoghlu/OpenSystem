/* foundry-log-model-private.h
 *
 * Copyright 2023-2024 Christian Hergert <chergert@redhat.com>
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

#include <glib-object.h>

G_BEGIN_DECLS

#define FOUNDRY_TYPE_LOG_MODEL (foundry_log_model_get_type())

G_DECLARE_FINAL_TYPE (FoundryLogModel, foundry_log_model, FOUNDRY, LOG_MODEL, GObject)

FoundryLogModel *_foundry_log_model_new        (void);
void             _foundry_log_model_remove_all (FoundryLogModel *self);
void             _foundry_log_model_append     (FoundryLogModel *model,
                                                const char      *domain,
                                                GLogLevelFlags   flags,
                                                char            *message);

G_END_DECLS
