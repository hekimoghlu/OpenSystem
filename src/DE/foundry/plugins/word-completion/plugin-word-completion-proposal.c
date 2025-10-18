/* plugin-word-completion-proposal.c
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

#include "plugin-word-completion-proposal.h"

struct _PluginWordCompletionProposal
{
  FoundryCompletionProposal  parent_instance;
  GRefString                *word;
  GRefString                *path;
};

enum {
  PROP_0,
  PROP_WORD,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (PluginWordCompletionProposal, plugin_word_completion_proposal, FOUNDRY_TYPE_COMPLETION_PROPOSAL)

static GParamSpec *properties[N_PROPS];
static GIcon *word_icon;

static GIcon *
plugin_word_completion_proposal_dup_icon (FoundryCompletionProposal *proposal)
{
  return g_object_ref (word_icon);
}

static char *
plugin_word_completion_proposal_dup_after (FoundryCompletionProposal *proposal)
{
  PluginWordCompletionProposal *self = PLUGIN_WORD_COMPLETION_PROPOSAL (proposal);

  return g_strdup (self->path);
}

static char *
plugin_word_completion_proposal_dup_typed_text (FoundryCompletionProposal *proposal)
{
  PluginWordCompletionProposal *self = PLUGIN_WORD_COMPLETION_PROPOSAL (proposal);

  return g_strdup (self->word);
}

static void
plugin_word_completion_proposal_finalize (GObject *object)
{
  PluginWordCompletionProposal *self = (PluginWordCompletionProposal *)object;

  g_clear_pointer (&self->word, g_ref_string_release);
  g_clear_pointer (&self->path, g_ref_string_release);

  G_OBJECT_CLASS (plugin_word_completion_proposal_parent_class)->finalize (object);
}

static void
plugin_word_completion_proposal_get_property (GObject    *object,
                                              guint       prop_id,
                                              GValue     *value,
                                              GParamSpec *pspec)
{
  PluginWordCompletionProposal *self = PLUGIN_WORD_COMPLETION_PROPOSAL (object);

  switch (prop_id)
    {
    case PROP_WORD:
      g_value_set_string (value, self->word);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_word_completion_proposal_class_init (PluginWordCompletionProposalClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryCompletionProposalClass *proposal_class = FOUNDRY_COMPLETION_PROPOSAL_CLASS (klass);

  object_class->finalize = plugin_word_completion_proposal_finalize;
  object_class->get_property = plugin_word_completion_proposal_get_property;

  proposal_class->dup_icon = plugin_word_completion_proposal_dup_icon;
  proposal_class->dup_typed_text = plugin_word_completion_proposal_dup_typed_text;
  proposal_class->dup_after = plugin_word_completion_proposal_dup_after;

  properties[PROP_WORD] =
    g_param_spec_string ("word", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  word_icon = g_themed_icon_new ("completion-word-symbolic");
}

static void
plugin_word_completion_proposal_init (PluginWordCompletionProposal *self)
{
}

PluginWordCompletionProposal *
plugin_word_completion_proposal_new (GRefString *word,
                                     GRefString *path)
{
  PluginWordCompletionProposal *self;

  self = g_object_new (PLUGIN_TYPE_WORD_COMPLETION_PROPOSAL, NULL);

  if (word != NULL)
    self->word = g_ref_string_acquire (word);

  if (path != NULL)
    self->path = g_ref_string_acquire (path);

  return self;
}
