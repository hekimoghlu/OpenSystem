/* foundry-action-responder.h
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

#include "foundry-responder-private.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_ACTION_RESPONDER (foundry_action_responder_get_type())

G_DECLARE_FINAL_TYPE (FoundryActionResponder, foundry_action_responder, FOUNDRY, ACTION_RESPONDER, FoundryResponder)

FoundryActionResponder *foundry_action_responder_new         (const char             *name,
                                                              const GVariantType     *parameter_type,
                                                              const GVariantType     *state_type);
gboolean                foundry_action_responder_get_enabled (FoundryActionResponder *self);
void                    foundry_action_responder_set_enabled (FoundryActionResponder *self,
                                                              gboolean                enabled);
GVariant               *foundry_action_responder_get_state   (FoundryActionResponder *self);
void                    foundry_action_responder_set_state   (FoundryActionResponder *self,
                                                              GVariant               *state);

G_END_DECLS
