/* foundry-action-responder-group-private.h
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

#include "foundry-action-responder-private.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_ACTION_RESPONDER_GROUP (foundry_action_responder_group_get_type())

G_DECLARE_FINAL_TYPE (FoundryActionResponderGroup, foundry_action_responder_group, FOUNDRY, ACTION_RESPONDER_GROUP, GObject)

FoundryActionResponderGroup *foundry_action_responder_group_new    (void);
void                         foundry_action_responder_group_add    (FoundryActionResponderGroup *self,
                                                                    FoundryActionResponder      *responder);
void                         foundry_action_responder_group_remove (FoundryActionResponderGroup *self,
                                                                    FoundryActionResponder      *responder);

G_END_DECLS
