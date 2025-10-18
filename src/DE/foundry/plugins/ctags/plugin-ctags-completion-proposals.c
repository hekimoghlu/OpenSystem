/* plugin-ctags-completion-proposals.c
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

#include "plugin-ctags-completion-proposal.h"
#include "plugin-ctags-completion-proposals.h"

struct _PluginCtagsCompletionProposals
{
  GObject parent_instance;
  PluginCtagsFile *file;
  EggBitset *bitset;
};

static GType
plugin_ctags_completion_proposals_get_item_type (GListModel *model)
{
  return G_TYPE_OBJECT;
}

static guint
plugin_ctags_completion_proposals_get_n_items (GListModel *model)
{
  return egg_bitset_get_size (PLUGIN_CTAGS_COMPLETION_PROPOSALS (model)->bitset);
}

static gpointer
plugin_ctags_completion_proposals_get_item (GListModel *model,
                                            guint       position)
{
  PluginCtagsCompletionProposals *self = PLUGIN_CTAGS_COMPLETION_PROPOSALS (model);
  guint index;

  if (position >= egg_bitset_get_size (self->bitset))
    return NULL;

  index = egg_bitset_get_nth (self->bitset, position);

  return plugin_ctags_completion_proposal_new (self->file, index);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = plugin_ctags_completion_proposals_get_item_type;
  iface->get_n_items = plugin_ctags_completion_proposals_get_n_items;
  iface->get_item = plugin_ctags_completion_proposals_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (PluginCtagsCompletionProposals, plugin_ctags_completion_proposals, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static void
plugin_ctags_completion_proposals_finalize (GObject *object)
{
  PluginCtagsCompletionProposals *self = (PluginCtagsCompletionProposals *)object;

  g_clear_object (&self->file);
  g_clear_pointer (&self->bitset, egg_bitset_unref);

  G_OBJECT_CLASS (plugin_ctags_completion_proposals_parent_class)->finalize (object);
}

static void
plugin_ctags_completion_proposals_class_init (PluginCtagsCompletionProposalsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = plugin_ctags_completion_proposals_finalize;
}

static void
plugin_ctags_completion_proposals_init (PluginCtagsCompletionProposals *self)
{
}

GListModel *
plugin_ctags_completion_proposals_new (PluginCtagsFile *file,
                                       EggBitset       *bitset)
{
  PluginCtagsCompletionProposals *self;

  g_return_val_if_fail (file != NULL, NULL);
  g_return_val_if_fail (bitset != NULL, NULL);

  self = g_object_new (PLUGIN_TYPE_CTAGS_COMPLETION_PROPOSALS, NULL);
  self->file = g_object_ref (file);
  self->bitset = egg_bitset_ref (bitset);

  return G_LIST_MODEL (self);
}
