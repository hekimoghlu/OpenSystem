/* foundry-source-completion-proposal.c
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

#include "foundry-source-completion-proposal-private.h"

struct _FoundrySourceCompletionProposal
{
  GObject parent_instance;
  FoundryCompletionProposal *proposal;
};

enum {
  PROP_0,
  PROP_PROPOSAL,
  N_PROPS
};

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundrySourceCompletionProposal, foundry_source_completion_proposal, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (GTK_SOURCE_TYPE_COMPLETION_PROPOSAL, NULL))

static GParamSpec *properties[N_PROPS];

static void
foundry_source_completion_proposal_dispose (GObject *object)
{
  FoundrySourceCompletionProposal *self = (FoundrySourceCompletionProposal *)object;

  g_clear_object (&self->proposal);

  G_OBJECT_CLASS (foundry_source_completion_proposal_parent_class)->dispose (object);
}

static void
foundry_source_completion_proposal_get_property (GObject    *object,
                                                 guint       prop_id,
                                                 GValue     *value,
                                                 GParamSpec *pspec)
{
  FoundrySourceCompletionProposal *self = FOUNDRY_SOURCE_COMPLETION_PROPOSAL (object);

  switch (prop_id)
    {
    case PROP_PROPOSAL:
      g_value_set_object (value, self->proposal);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_source_completion_proposal_set_property (GObject      *object,
                                                 guint         prop_id,
                                                 const GValue *value,
                                                 GParamSpec   *pspec)
{
  FoundrySourceCompletionProposal *self = FOUNDRY_SOURCE_COMPLETION_PROPOSAL (object);

  switch (prop_id)
    {
    case PROP_PROPOSAL:
      self->proposal = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_source_completion_proposal_class_init (FoundrySourceCompletionProposalClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_source_completion_proposal_dispose;
  object_class->get_property = foundry_source_completion_proposal_get_property;
  object_class->set_property = foundry_source_completion_proposal_set_property;

  properties[PROP_PROPOSAL] =
    g_param_spec_object ("proposal", NULL, NULL,
                         FOUNDRY_TYPE_COMPLETION_PROPOSAL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_source_completion_proposal_init (FoundrySourceCompletionProposal *self)
{
}

FoundrySourceCompletionProposal *
foundry_source_completion_proposal_new (FoundryCompletionProposal *proposal)
{
  g_return_val_if_fail (FOUNDRY_IS_COMPLETION_PROPOSAL (proposal), NULL);

  return g_object_new (FOUNDRY_TYPE_SOURCE_COMPLETION_PROPOSAL,
                       "proposal", proposal,
                       NULL);
}

FoundryCompletionProposal *
foundry_source_completion_proposal_get_proposal (FoundrySourceCompletionProposal *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SOURCE_COMPLETION_PROPOSAL (self), NULL);

  return self->proposal;
}
