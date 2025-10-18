/* foundry-lsp-completion-results.c
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

#include "eggbitset.h"

#include <gio/gio.h>

#include "foundry-lsp-client.h"
#include "foundry-lsp-completion-proposal-private.h"
#include "foundry-lsp-completion-results-private.h"
#include "foundry-util.h"

#define EGG_ARRAY_NAME items
#define EGG_ARRAY_TYPE_NAME Items
#define EGG_ARRAY_ELEMENT_TYPE gpointer
#define EGG_ARRAY_BY_VALUE 1
#include "eggarrayimpl.c"

struct _FoundryLspCompletionResults
{
  GObject           parent_instance;
  FoundryLspClient *client;
  JsonNode         *reply;
  JsonNode         *results;
  EggBitset        *bitset;
  char             *typed_text;
  GQueue            children;
  Items             items;
};

enum {
  PROP_0,
  PROP_CLIENT,
  N_PROPS
};

static GType
foundry_lsp_completion_results_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_COMPLETION_PROPOSAL;
}

static guint
foundry_lsp_completion_results_get_n_items (GListModel *model)
{
  FoundryLspCompletionResults *self = FOUNDRY_LSP_COMPLETION_RESULTS (model);

  return egg_bitset_get_size (self->bitset);
}

static gpointer
foundry_lsp_completion_results_get_item (GListModel *model,
                                         guint       position)
{
  FoundryLspCompletionResults *self = FOUNDRY_LSP_COMPLETION_RESULTS (model);
  gpointer *item;
  gsize index;

  if (position >= egg_bitset_get_size (self->bitset))
    return NULL;

  index = egg_bitset_get_nth (self->bitset, position);
  item = items_get (&self->items, index);

  if (*item == NULL)
    {
      JsonArray *ar = json_node_get_array (self->results);
      JsonNode *node = json_array_get_element (ar, index);
      FoundryLspCompletionProposal *proposal;

      proposal = _foundry_lsp_completion_proposal_new (node);
      proposal->container = self;
      proposal->indexed = item;
      g_queue_push_tail_link (&self->children, &proposal->link);

      *item = proposal;

      return g_steal_pointer (&proposal);
    }

  return g_object_ref (*item);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_lsp_completion_results_get_item_type;
  iface->get_n_items = foundry_lsp_completion_results_get_n_items;
  iface->get_item = foundry_lsp_completion_results_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryLspCompletionResults, foundry_lsp_completion_results, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static GParamSpec *properties[N_PROPS];

static void
foundry_lsp_completion_results_dispose (GObject *object)
{
  FoundryLspCompletionResults *self = (FoundryLspCompletionResults *)object;

  egg_bitset_remove_all (self->bitset);

  while (self->children.head != NULL)
    foundry_lsp_completion_results_unlink (self, FOUNDRY_LSP_COMPLETION_PROPOSAL (g_queue_peek_head (&self->children)));

  items_clear (&self->items);

  g_clear_pointer (&self->typed_text, g_free);
  g_clear_object (&self->client);
  g_clear_pointer (&self->reply, json_node_unref);
  g_clear_pointer (&self->results, json_node_unref);

  G_OBJECT_CLASS (foundry_lsp_completion_results_parent_class)->dispose (object);
}

static void
foundry_lsp_completion_results_finalize (GObject *object)
{
  FoundryLspCompletionResults *self = (FoundryLspCompletionResults *)object;

  g_clear_pointer (&self->bitset, egg_bitset_unref);

  G_OBJECT_CLASS (foundry_lsp_completion_results_parent_class)->finalize (object);
}

static void
foundry_lsp_completion_results_get_property (GObject    *object,
                                             guint       prop_id,
                                             GValue     *value,
                                             GParamSpec *pspec)
{
  FoundryLspCompletionResults *self = FOUNDRY_LSP_COMPLETION_RESULTS (object);

  switch (prop_id)
    {
    case PROP_CLIENT:
      g_value_set_object (value, self->client);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_lsp_completion_results_set_property (GObject      *object,
                                             guint         prop_id,
                                             const GValue *value,
                                             GParamSpec   *pspec)
{
  FoundryLspCompletionResults *self = FOUNDRY_LSP_COMPLETION_RESULTS (object);

  switch (prop_id)
    {
    case PROP_CLIENT:
      self->client = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_lsp_completion_results_class_init (FoundryLspCompletionResultsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_lsp_completion_results_dispose;
  object_class->finalize = foundry_lsp_completion_results_finalize;
  object_class->get_property = foundry_lsp_completion_results_get_property;
  object_class->set_property = foundry_lsp_completion_results_set_property;

  properties[PROP_CLIENT] =
    g_param_spec_object ("client", NULL, NULL,
                         FOUNDRY_TYPE_LSP_CLIENT,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_lsp_completion_results_init (FoundryLspCompletionResults *self)
{
  self->bitset = egg_bitset_new_empty ();
}

/**
 * foundry_lsp_completion_results_dup_client:
 * @self: a [class@Foundry.LspCompletionResults]
 *
 * Returns: (transfer full): a [class@Foundry.CompletionResults]
 */
FoundryLspClient *
foundry_lsp_completion_results_dup_client (FoundryLspCompletionResults *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LSP_COMPLETION_RESULTS (self), NULL);

  return g_object_ref (self->client);
}

static DexFuture *
foundry_lsp_completion_results_load (FoundryLspCompletionResults *self,
                                     const char                  *typed_text)
{
  gsize n_children;

  g_assert (FOUNDRY_IS_LSP_COMPLETION_RESULTS (self));

  n_children = json_array_get_length (json_node_get_array (self->results));

  items_set_size (&self->items, n_children);

  if (n_children > 0)
    egg_bitset_add_range (self->bitset, 0, n_children);

  foundry_lsp_completion_results_refilter (self, typed_text);

  return dex_future_new_take_object (g_object_ref (self));
}

/**
 * foundry_lsp_completion_results_new:
 * @client: a [class@Foundry.LspClient]
 * @reply: the reply from the LSP-enabled server
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@Foundry.LspCompletionResults].
 */
DexFuture *
foundry_lsp_completion_results_new (FoundryLspClient *client,
                                    JsonNode         *reply,
                                    const char       *typed_text)
{
  g_autoptr(FoundryLspCompletionResults) self = NULL;
  JsonObject *obj;
  JsonNode *items;

  dex_return_error_if_fail (FOUNDRY_IS_LSP_CLIENT (client));
  dex_return_error_if_fail (reply != NULL);

  /* Possibly unwrap the {items: []} style result. */
  if (JSON_NODE_HOLDS_OBJECT (reply) &&
      (obj = json_node_get_object (reply)) &&
      json_object_has_member (obj, "items") &&
      (items = json_object_get_member (obj, "items")) &&
      JSON_NODE_HOLDS_ARRAY (items))
    {
      self = g_object_new (FOUNDRY_TYPE_LSP_COMPLETION_RESULTS,
                           "client", client,
                           NULL);
      self->reply = json_node_ref (reply);
      self->results = json_node_ref (items);

      return foundry_scheduler_spawn (dex_thread_pool_scheduler_get_default (), 0,
                                      G_CALLBACK (foundry_lsp_completion_results_load),
                                      2,
                                      FOUNDRY_TYPE_LSP_COMPLETION_RESULTS, self,
                                      G_TYPE_STRING, typed_text);
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_INVALID_DATA,
                                "Invalid completion reply from peer");
}

typedef enum _Change
{
  SAME,
  DIFFERENT,
  MORE_STRICT,
  LESS_STRICT,
} Change;

static Change
determine_change (const char *before,
                  const char *after)
{
  if (before == after)
    return SAME;

  if (g_strcmp0 (before, after) == 0)
    return SAME;

  if (before == NULL || after == NULL)
    return DIFFERENT;

  if (before[0] == 0)
    return MORE_STRICT;

  if (after[0] == 0)
    return LESS_STRICT;

  if (g_str_has_prefix (after, before))
    return MORE_STRICT;

  if (g_str_has_prefix (before, after))
    return LESS_STRICT;

  return DIFFERENT;
}

static gboolean
fuzzy_match (const char *haystack,
             const char *casefold_needle,
             guint      *priority)
{
	int real_score = 0;

  if (haystack == NULL || haystack[0] == 0)
    return FALSE;

  for (; *casefold_needle; casefold_needle = g_utf8_next_char (casefold_needle))
    {
      gunichar ch = g_utf8_get_char (casefold_needle);
      gunichar chup = g_unichar_toupper (ch);
      const gchar *tmp;
      const gchar *downtmp;
      const gchar *uptmp;

      /*
       * Note that the following code is not really correct. We want
       * to be relatively fast here, but we also don't want to convert
       * strings to casefolded versions for querying on each compare.
       * So we use the casefold version and compare with upper. This
       * works relatively well since we are usually dealing with ASCII
       * for function names and symbols.
       */

      downtmp = strchr (haystack, ch);
      uptmp = strchr (haystack, chup);

      if (downtmp && uptmp)
        tmp = MIN (downtmp, uptmp);
      else if (downtmp)
        tmp = downtmp;
      else if (uptmp)
        tmp = uptmp;
      else
        return FALSE;

      /*
       * Here we calculate the cost of this character into the score.
       * If we matched exactly on the next character, the cost is ZERO.
       * However, if we had to skip some characters, we have a cost
       * of 2*distance to the character. This is necessary so that
       * when we add the cost of the remaining haystack, strings which
       * exhausted @casefold_needle score lower (higher priority) than
       * strings which had to skip characters but matched the same
       * number of characters in the string.
       */
      real_score += (tmp - haystack) * 2;

      /* Add extra cost if we matched by using toupper */
      if ((gunichar)*haystack == chup)
        real_score += 1;

      /*
       * * Now move past our matching character so we cannot match
       * * it a second time.
       * */
      haystack = tmp + 1;
    }

  if (priority != NULL)
    *priority = real_score + strlen (haystack);

  return TRUE;
}

void
foundry_lsp_completion_results_refilter (FoundryLspCompletionResults *self,
                                         const char                  *typed_text)
{
  guint old_n_items;
  guint new_n_items;

  g_return_if_fail (FOUNDRY_IS_LSP_COMPLETION_RESULTS (self));

  old_n_items = g_list_model_get_n_items (G_LIST_MODEL (self));

  switch (determine_change (self->typed_text, typed_text))
    {
    case SAME:
      return;

    case DIFFERENT:
      egg_bitset_remove_all (self->bitset);
      egg_bitset_add_range (self->bitset, 0, items_get_size (&self->items));
      G_GNUC_FALLTHROUGH;

    case MORE_STRICT:
      {
        EggBitsetIter iter;
        guint index;

        if (typed_text != NULL &&
            typed_text[0] != 0 &&
            egg_bitset_iter_init_first (&iter, self->bitset, &index))
          {
            g_autofree char *casefold = g_utf8_casefold (typed_text, -1);
            JsonArray *ar = json_node_get_array (self->results);

            do
              {
                JsonNode *child = json_array_get_element (ar, index);
                const char *label;
                JsonObject *obj;

                if (!(JSON_NODE_HOLDS_OBJECT (child) &&
                      (obj = json_node_get_object (child)) &&
                      json_object_has_member (obj, "label") &&
                      (label = json_object_get_string_member (obj, "label")) &&
                      fuzzy_match (label, casefold, NULL)))
                  egg_bitset_remove (self->bitset, index);
              }
            while (egg_bitset_iter_next (&iter, &index));
          }
      }
      break;

    case LESS_STRICT:
      {
        if (typed_text == NULL || typed_text[0] != 0)
          {
            egg_bitset_remove_all (self->bitset);
            egg_bitset_add_range (self->bitset, 0, items_get_size (&self->items));
          }
        else
          {
            g_autoptr(EggBitset) other = NULL;
            EggBitsetIter iter;
            guint index;

            other = egg_bitset_new_empty ();
            egg_bitset_add_range (other, 0, items_get_size (&self->items));
            egg_bitset_difference (other, self->bitset);

            if (egg_bitset_iter_init_first (&iter, other, &index))
              {
                g_autofree char *casefold = g_utf8_casefold (typed_text, -1);
                JsonArray *ar = json_node_get_array (self->results);

                do
                  {
                    JsonNode *child = json_array_get_element (ar, index);
                    const char *label;
                    JsonObject *obj;

                    g_assert (!egg_bitset_contains (self->bitset, index));

                    if (JSON_NODE_HOLDS_OBJECT (child) &&
                        (obj = json_node_get_object (child)) &&
                        json_object_has_member (obj, "label") &&
                        (label = json_object_get_string_member (obj, "label")) &&
                        fuzzy_match (label, casefold, NULL))
                      egg_bitset_add (self->bitset, index);
                  }
                while (egg_bitset_iter_next (&iter, &index));
              }
          }
      }
      break;

    default:
      g_assert_not_reached ();
    }

  g_set_str (&self->typed_text, typed_text);

  new_n_items = g_list_model_get_n_items (G_LIST_MODEL (self));

  g_list_model_items_changed (G_LIST_MODEL (self), 0, old_n_items, new_n_items);
}

void
foundry_lsp_completion_results_unlink (FoundryLspCompletionResults  *self,
                                       FoundryLspCompletionProposal *proposal)
{
  g_assert (FOUNDRY_IS_LSP_COMPLETION_RESULTS (self));
  g_assert (FOUNDRY_IS_LSP_COMPLETION_PROPOSAL (proposal));

  proposal->container = NULL;

  *proposal->indexed = NULL;
  proposal->indexed = NULL;

  g_queue_unlink (&self->children, &proposal->link);
}
