/* plugin-word-completion-results.c
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

#include "line-reader-private.h"

#include "plugin-word-completion-proposal.h"
#include "plugin-word-completion-results.h"

#define WORD_MIN 3
#define MAX_ITEMS 10000
#define MAX_DEPTH 3
#define _1_MSEC (G_USEC_PER_SEC/1000)

typedef struct _Proposal
{
  GRefString *word;
  GRefString *path;
} Proposal;

struct _PluginWordCompletionResults
{
  GObject     parent_instance;
  GBytes     *bytes;
  DexFuture  *future;
  GSequence  *sequence;
  GHashTable *seen_files;
  char       *language_id;
  GFile      *file;
  GFile      *dir;
  gint64      next_deadline;
  guint       cached_size;
};

static GType
plugin_word_completion_results_get_item_type (GListModel *model)
{
  return PLUGIN_TYPE_WORD_COMPLETION_PROPOSAL;
}

static guint
plugin_word_completion_results_get_n_items (GListModel *model)
{
  PluginWordCompletionResults *self = PLUGIN_WORD_COMPLETION_RESULTS (model);

  return self->cached_size;
}

static gpointer
plugin_word_completion_results_get_item (GListModel *model,
                                         guint       position)
{
  PluginWordCompletionResults *self = PLUGIN_WORD_COMPLETION_RESULTS (model);
  GSequenceIter *iter;
  Proposal *proposal;

  iter = g_sequence_get_iter_at_pos (self->sequence, position);

  if (g_sequence_iter_is_end (iter))
    return NULL;

  proposal = g_sequence_get (iter);

  return plugin_word_completion_proposal_new (proposal->word, proposal->path);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = plugin_word_completion_results_get_item_type;
  iface->get_n_items = plugin_word_completion_results_get_n_items;
  iface->get_item = plugin_word_completion_results_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (PluginWordCompletionResults, plugin_word_completion_results, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static GRegex *regex;
static GRegex *include_regex;
static const char *include_languages[] = { "c", "cpp", "chdr", "cpphdr", "objc", NULL };

static void
proposal_free (Proposal *proposal)
{
  g_clear_pointer (&proposal->word, g_ref_string_release);
  g_clear_pointer (&proposal->path, g_ref_string_release);
  g_free (proposal);
}

static int
proposal_compare (gconstpointer a,
                  gconstpointer b)
{
  const Proposal *prop_a = a;
  const Proposal *prop_b = b;

  return strcmp (prop_a->word, prop_b->word);
}

static gboolean
proposal_matches (const Proposal *prop,
                  const char     *word)
{
  return strcmp (prop->word, word) == 0;
}

static void
plugin_word_completion_results_add (PluginWordCompletionResults *self,
                                    GFile                       *file,
                                    const char                  *word)
{
  GSequenceIter *iter;
  Proposal *proposal;
  Proposal lookup = {(char *)word, NULL};

  g_assert (PLUGIN_IS_WORD_COMPLETION_RESULTS (self));
  g_assert (!file || G_IS_FILE (file));
  g_assert (word != NULL);

  iter = g_sequence_search (self->sequence,
                            &lookup,
                            (GCompareDataFunc)proposal_compare,
                            NULL);

  if (!g_sequence_iter_is_end (iter))
    {
      GSequenceIter *prev;

      if (proposal_matches (g_sequence_get (iter), word))
        return;

      prev = g_sequence_iter_prev (iter);

      if (prev != iter && proposal_matches (g_sequence_get (prev), word))
        return;
    }

  proposal = g_new0 (Proposal, 1);
  proposal->word = g_ref_string_new (word);

  if (file != NULL && self->file != NULL && !g_file_equal (self->file, file))
    {
      GRefString *refstr;

      if (!(refstr = g_hash_table_lookup (self->seen_files, file)))
        {
          if (self->dir != NULL && g_file_has_prefix (file, self->dir))
            {
              g_autofree char *relative_path = g_file_get_relative_path (self->dir, file);
              refstr = g_ref_string_new (relative_path);
            }
          else
            {
              g_autofree char *path = g_file_get_path (file);
              refstr = g_ref_string_new (path);
            }

          g_hash_table_replace (self->seen_files, g_object_ref (file), refstr);
        }

      proposal->path = g_ref_string_acquire (refstr);
    }

  iter = g_sequence_insert_before (iter, proposal);

  self->cached_size++;

  g_list_model_items_changed (G_LIST_MODEL (self),
                              g_sequence_iter_get_position (iter),
                              0, 1);
}

static void
clear_refstr (GRefString *str)
{
  if (str != NULL)
    g_ref_string_release (str);
}

static void
plugin_word_completion_results_finalize (GObject *object)
{
  PluginWordCompletionResults *self = (PluginWordCompletionResults *)object;

  g_clear_pointer (&self->bytes, g_bytes_unref);
  g_clear_pointer (&self->sequence, g_sequence_free);
  g_clear_pointer (&self->language_id, g_free);
  g_clear_pointer (&self->seen_files, g_hash_table_unref);
  g_clear_object (&self->file);
  g_clear_object (&self->dir);

  dex_clear (&self->future);

  G_OBJECT_CLASS (plugin_word_completion_results_parent_class)->finalize (object);
}

static void
plugin_word_completion_results_class_init (PluginWordCompletionResultsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = plugin_word_completion_results_finalize;

  regex = g_regex_new ("\\w+", G_REGEX_OPTIMIZE, G_REGEX_MATCH_DEFAULT, NULL);
  include_regex = g_regex_new ("^\\s*#\\s*(include|import)\\s*[\"<.]+(.+)[\">]\\s*$", G_REGEX_OPTIMIZE, G_REGEX_MATCH_DEFAULT, NULL);
}

static void
plugin_word_completion_results_init (PluginWordCompletionResults *self)
{
  self->sequence = g_sequence_new ((GDestroyNotify) proposal_free);
  self->seen_files = g_hash_table_new_full (g_file_hash,
                                            (GEqualFunc) g_file_equal,
                                            g_object_unref,
                                            (GDestroyNotify) clear_refstr);
}

PluginWordCompletionResults *
plugin_word_completion_results_new (GFile      *file,
                                    GBytes     *bytes,
                                    const char *language_id)
{
  PluginWordCompletionResults *self;

  g_return_val_if_fail (!file || G_IS_FILE (file), NULL);
  g_return_val_if_fail (bytes != NULL, NULL);

  self = g_object_new (PLUGIN_TYPE_WORD_COMPLETION_RESULTS, NULL);
  self->bytes = g_bytes_ref (bytes);
  self->language_id = g_strdup (language_id);

  if (file != NULL)
    {
      self->file = g_object_ref (file);
      self->dir = g_file_get_parent (file);

      g_hash_table_insert (self->seen_files, g_object_ref (file), NULL);
    }

  return self;
}

static void
plugin_word_completion_results_mine (PluginWordCompletionResults *self,
                                     GFile                       *file,
                                     GBytes                      *bytes,
                                     guint                        depth,
                                     gboolean                     follow_includes)
{
  g_autoptr(GFile) dir = NULL;
  LineReader reader;
  const char *line;
  gsize line_len;

  g_assert (PLUGIN_IS_WORD_COMPLETION_RESULTS (self));
  g_assert (!file || G_IS_FILE (file));
  g_assert (bytes != NULL);

  if (depth > MAX_DEPTH || file == NULL)
    follow_includes = FALSE;

  line_reader_init_from_bytes (&reader, bytes);

  while ((line = line_reader_next (&reader, &line_len)))
    {
      g_autoptr(GMatchInfo) match_info = NULL;
      gint64 now;

      if (line_len < WORD_MIN)
        continue;

      if (follow_includes && line_len >= 12)
        {
          g_autoptr(GMatchInfo) include_matches = NULL;

          if (g_regex_match_full (include_regex, line, line_len, 0, G_REGEX_MATCH_DEFAULT, &include_matches, NULL) &&
              g_match_info_matches (include_matches))
            {
              g_autofree char *word = g_match_info_fetch (include_matches, 2);

              if (!foundry_str_empty0 (word))
                {
                  g_autoptr(GFile) child = NULL;
                  g_autoptr(GBytes) child_bytes = NULL;

                  if (dir == NULL)
                    dir = g_file_get_parent (file);

                  child = g_file_get_child (dir, word);

                  if (!g_hash_table_contains (self->seen_files, child) &&
                      (child_bytes = dex_await_boxed (dex_file_load_contents_bytes (child), NULL)))
                    {
                      g_hash_table_insert (self->seen_files, g_object_ref (child), NULL);
                      plugin_word_completion_results_mine (self, child, child_bytes, depth + 1, follow_includes);
                    }
                }
            }
        }

      if (g_regex_match_full (regex, line, line_len, 0, G_REGEX_MATCH_DEFAULT, &match_info, NULL))
        {
          if (g_match_info_matches (match_info))
            {
              do
                {
                  g_autofree char *word = g_match_info_fetch (match_info, 0);

                  if (strlen (word) < WORD_MIN)
                    continue;

                  plugin_word_completion_results_add (self, file, word);

                  if (self->cached_size > MAX_ITEMS)
                    return;
                }
              while (g_match_info_next (match_info, NULL));
            }
        }

      now = g_get_monotonic_time ();

      if (now > self->next_deadline)
        {
          dex_await (dex_timeout_new_deadline (now + _1_MSEC), NULL);
          self->next_deadline = g_get_monotonic_time () + _1_MSEC;
        }
    }
}

static DexFuture *
plugin_word_completion_results_fiber (gpointer data)
{
  PluginWordCompletionResults *self = data;
  gboolean check_includes;

  g_assert (PLUGIN_IS_WORD_COMPLETION_RESULTS (self));
  g_assert (self->bytes != NULL);

  self->next_deadline = g_get_monotonic_time () + _1_MSEC;

  check_includes = (self->file != NULL &&
                    self->language_id != NULL &&
                    g_strv_contains (include_languages, self->language_id));

  plugin_word_completion_results_mine (self, self->file, self->bytes, 0, check_includes);

  return dex_future_new_true ();
}

DexFuture *
plugin_word_completion_results_await (PluginWordCompletionResults *self)
{
  g_return_val_if_fail (PLUGIN_IS_WORD_COMPLETION_RESULTS (self), NULL);

  if (self->future == NULL)
    self->future = dex_scheduler_spawn (NULL, 0,
                                        plugin_word_completion_results_fiber,
                                        g_object_ref (self),
                                        g_object_unref);

  return dex_ref (self->future);
}
