/* foundry-property-reaction.h
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

#include <gtk/gtk.h>

#include "foundry-reaction-private.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_PROPERTY_REACTION (foundry_property_reaction_get_type())

G_DECLARE_FINAL_TYPE (FoundryPropertyReaction, foundry_property_reaction, FOUNDRY, PROPERTY_REACTION, FoundryReaction)

FoundryPropertyReaction *foundry_property_reaction_new        (GObject                 *object,
                                                               const char              *property,
                                                               GtkExpression           *value);
void                     foundry_property_reaction_set_object (FoundryPropertyReaction *self,
                                                               GObject                 *object);
GObject                 *foundry_property_reaction_dup_object (FoundryPropertyReaction *self);
const char              *foundry_property_reaction_get_name   (FoundryPropertyReaction *self);
void                     foundry_property_reaction_set_name   (FoundryPropertyReaction *self,
                                                               const char              *name);
GtkExpression           *foundry_property_reaction_get_value  (FoundryPropertyReaction *self);
void                     foundry_property_reaction_set_value  (FoundryPropertyReaction *self,
                                                               GtkExpression           *value);

G_END_DECLS
