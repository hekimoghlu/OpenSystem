/* foundry-reaction.c
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

#include "foundry-reaction-private.h"

G_DEFINE_ABSTRACT_TYPE (FoundryReaction, foundry_reaction, G_TYPE_OBJECT)

static void
foundry_reaction_class_init (FoundryReactionClass *klass)
{
}

static void
foundry_reaction_init (FoundryReaction *self)
{
}

void
foundry_reaction_react (FoundryReaction *self)
{
  g_return_if_fail (FOUNDRY_IS_REACTION (self));

  return FOUNDRY_REACTION_GET_CLASS (self)->react (self);
}
