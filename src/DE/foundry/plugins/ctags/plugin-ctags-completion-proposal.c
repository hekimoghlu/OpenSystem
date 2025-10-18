/* plugin-ctags-completion-proposal.c
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

struct _PluginCtagsCompletionProposal
{
  FoundryCompletionProposal parent_instance;
  PluginCtagsFile *file;
  guint position;
};

static GIcon *class_icon;
static GIcon *enum_icon;
static GIcon *function_icon;
static GIcon *macro_icon;
static GIcon *member_icon;
static GIcon *struct_icon;

G_DEFINE_FINAL_TYPE (PluginCtagsCompletionProposal, plugin_ctags_completion_proposal, FOUNDRY_TYPE_COMPLETION_PROPOSAL)

static char *
plugin_ctags_completion_proposal_dup_typed_text (FoundryCompletionProposal *proposal)
{
  PluginCtagsCompletionProposal *self = PLUGIN_CTAGS_COMPLETION_PROPOSAL (proposal);

  return plugin_ctags_file_dup_name (self->file, self->position);
}

static GIcon *
plugin_ctags_completion_proposal_dup_icon (FoundryCompletionProposal *proposal)
{
  PluginCtagsCompletionProposal *self = PLUGIN_CTAGS_COMPLETION_PROPOSAL (proposal);
  PluginCtagsKind kind = plugin_ctags_file_get_kind (self->file, self->position);

  switch ((int)kind)
    {
    case PLUGIN_CTAGS_KIND_CLASS_NAME:
      return g_object_ref (class_icon);

    case PLUGIN_CTAGS_KIND_PROTOTYPE:
    case PLUGIN_CTAGS_KIND_FUNCTION:
      return g_object_ref (function_icon);

    case PLUGIN_CTAGS_KIND_DEFINE:
      return g_object_ref (macro_icon);

    case PLUGIN_CTAGS_KIND_STRUCTURE:
      return g_object_ref (struct_icon);

    case PLUGIN_CTAGS_KIND_ENUMERATION_NAME:
    case PLUGIN_CTAGS_KIND_ENUMERATOR:
      return g_object_ref (enum_icon);

    default:
      return NULL;
    }
}

static void
plugin_ctags_completion_proposal_finalize (GObject *object)
{
  PluginCtagsCompletionProposal *self = (PluginCtagsCompletionProposal *)object;

  g_clear_object (&self->file);

  G_OBJECT_CLASS (plugin_ctags_completion_proposal_parent_class)->finalize (object);
}

static void
plugin_ctags_completion_proposal_class_init (PluginCtagsCompletionProposalClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryCompletionProposalClass *proposal_class = FOUNDRY_COMPLETION_PROPOSAL_CLASS (klass);

  object_class->finalize = plugin_ctags_completion_proposal_finalize;

  proposal_class->dup_icon = plugin_ctags_completion_proposal_dup_icon;
  proposal_class->dup_typed_text = plugin_ctags_completion_proposal_dup_typed_text;
}

static void
plugin_ctags_completion_proposal_init (PluginCtagsCompletionProposal *self)
{
  if (class_icon == NULL)
    class_icon = g_themed_icon_new ("lang-class-symbolic");

  if (function_icon == NULL)
    function_icon = g_themed_icon_new ("lang-function-symbolic");

  if (macro_icon == NULL)
    macro_icon = g_themed_icon_new ("lang-macro-symbolic");

  if (struct_icon == NULL)
    struct_icon = g_themed_icon_new ("lang-struct-symbolic");

  if (member_icon == NULL)
    member_icon = g_themed_icon_new ("lang-struct-field-symbolic");

  if (enum_icon == NULL)
    enum_icon = g_themed_icon_new ("lang-enum-symbolic");
}

PluginCtagsCompletionProposal *
plugin_ctags_completion_proposal_new (PluginCtagsFile *file,
                                      guint            position)
{
  PluginCtagsCompletionProposal *self;

  self = g_object_new (PLUGIN_TYPE_CTAGS_COMPLETION_PROPOSAL, NULL);
  self->file = g_object_ref (file);
  self->position = position;

  return self;
}
