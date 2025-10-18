/* foundry-multi-reaction.c
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

#include "foundry-multi-reaction-private.h"

struct _FoundryMultiReaction
{
  FoundryReaction  parent_instance;
  GPtrArray       *reactions;
};

static void
foundry_multi_reaction_add_child (GtkBuildable *buildable,
                                  GtkBuilder   *builder,
                                  GObject      *object,
                                  const char   *name)
{
  FoundryMultiReaction *self = FOUNDRY_MULTI_REACTION (buildable);

  if (FOUNDRY_IS_REACTION (object))
    foundry_multi_reaction_add (self, g_object_ref (FOUNDRY_REACTION (object)));
  else
    g_critical ("Cannot add child typed %s to %s",
                G_OBJECT_TYPE_NAME (object),
                G_OBJECT_TYPE_NAME (self));
}

static void
buildable_iface_init (GtkBuildableIface *iface)
{
  iface->add_child = foundry_multi_reaction_add_child;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryMultiReaction, foundry_multi_reaction, FOUNDRY_TYPE_REACTION,
                               G_IMPLEMENT_INTERFACE (GTK_TYPE_BUILDABLE, buildable_iface_init))

static void
foundry_multi_reaction_react (FoundryReaction *reaction)
{
  FoundryMultiReaction *self = FOUNDRY_MULTI_REACTION (reaction);

  for (guint i = 0; i < self->reactions->len; i++)
    foundry_reaction_react (g_ptr_array_index (self->reactions, i));
}

static void
foundry_multi_reaction_finalize (GObject *object)
{
  FoundryMultiReaction *self = (FoundryMultiReaction *)object;

  g_clear_pointer (&self->reactions, g_ptr_array_unref);

  G_OBJECT_CLASS (foundry_multi_reaction_parent_class)->finalize (object);
}

static void
foundry_multi_reaction_class_init (FoundryMultiReactionClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryReactionClass *reaction_class = FOUNDRY_REACTION_CLASS (klass);

  object_class->finalize = foundry_multi_reaction_finalize;

  reaction_class->react = foundry_multi_reaction_react;
}

static void
foundry_multi_reaction_init (FoundryMultiReaction *self)
{
  self->reactions = g_ptr_array_new_with_free_func (g_object_unref);
}

FoundryMultiReaction *
foundry_multi_reaction_new (void)
{
  return g_object_new (FOUNDRY_TYPE_MULTI_REACTION, NULL);
}

/**
 * foundry_multi_reaction_add:
 * @self: a [class@Foundry.MultiReaction]
 * @reaction: (transfer full): a [class@Foundry.Reaction]
 *
 * Adds @reaction to the list of reactions to be performed
 * when @self reacts.
 */
void
foundry_multi_reaction_add (FoundryMultiReaction *self,
                            FoundryReaction      *reaction)
{
  g_return_if_fail (FOUNDRY_IS_MULTI_REACTION (self));
  g_return_if_fail (FOUNDRY_IS_REACTION (reaction));

  g_ptr_array_add (self->reactions, reaction);
}

/**
 * foundry_multi_reaction_remove:
 * @self: a [class@Foundry.MultiReaction]
 * @reaction: a [class@Foundry.Reaction] previously added
 *
 * Removes @reaction from the list of [class@Foundry.Reaction] that will
 * react in response to @self reacting.
 */
void
foundry_multi_reaction_remove (FoundryMultiReaction *self,
                               FoundryReaction      *reaction)
{
  g_return_if_fail (FOUNDRY_IS_MULTI_REACTION (self));
  g_return_if_fail (FOUNDRY_IS_REACTION (reaction));

  if (!g_ptr_array_remove (self->reactions, reaction))
    g_warning ("%s at %p is missing from %s",
               G_OBJECT_TYPE_NAME (reaction),
               reaction,
               G_OBJECT_TYPE_NAME (self));
}
