/* spelling-engine.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include <gtksourceview/gtksource.h>

#include "cjhtextregionprivate.h"

#include "spelling-engine-private.h"
#include "spelling-job-private.h"

#define TAG_NEEDS_CHECK        GUINT_TO_POINTER(1)
#define TAG_CHECKED            GUINT_TO_POINTER(0)
#define INVALIDATE_DELAY_MSECS 100
#define WATERMARK_PER_JOB      1000

struct _SpellingEngine
{
  GObject          parent_instance;
  CjhTextRegion   *region;
  GWeakRef         instance_wr;
  SpellingJob     *active;
  SpellingAdapter  adapter;
  guint            queued_update_handler;
};

typedef struct
{
  SpellingEngine *self;
  GObject        *instance;
  GtkBitset      *bitset;
  GtkBitset      *all;
  guint           size;
} CollectRanges;

G_DEFINE_FINAL_TYPE (SpellingEngine, spelling_engine, G_TYPE_OBJECT)

static void spelling_engine_queue_update (SpellingEngine *self,
                                          guint           delay_msec);

static gboolean
spelling_engine_check_enabled (SpellingEngine *self)
{
  g_autoptr(GObject) instance = g_weak_ref_get (&self->instance_wr);

  if (instance != NULL)
    return self->adapter.check_enabled (instance);

  return FALSE;
}

static gboolean
has_unchecked_regions_cb (gsize                   offset,
                          const CjhTextRegionRun *run,
                          gpointer                user_data)
{
  gboolean *ret = user_data;
  *ret |= run->data == TAG_NEEDS_CHECK;
  return *ret;
}

static gboolean
spelling_engine_has_unchecked_regions (SpellingEngine *self)
{
  gboolean ret = FALSE;
  _cjh_text_region_foreach (self->region, has_unchecked_regions_cb, &ret);
  return ret;
}

static gboolean
spelling_engine_extend_range (SpellingEngine *self,
                              guint          *begin,
                              guint          *end)
{
  g_autoptr(GObject) instance = NULL;

  g_assert (SPELLING_IS_ENGINE (self));
  g_assert (begin != NULL);
  g_assert (end != NULL);

  if ((instance = g_weak_ref_get (&self->instance_wr)))
    {
      guint tmp;

      tmp = *begin;
      if (self->adapter.backward_word_start (instance, &tmp))
        *begin = tmp;

      tmp = *end;
      if (self->adapter.forward_word_end (instance, &tmp))
        *end = tmp;
    }

  return *begin != *end;
}

static void
spelling_engine_add_fragment (SpellingEngine *self,
                              GObject        *instance,
                              SpellingJob    *job,
                              GtkBitset      *bitset,
                              guint           begin,
                              guint           end)
{
  g_autoptr(GBytes) bytes = NULL;
  char *text;

  g_assert (SPELLING_IS_ENGINE (self));
  g_assert (bitset != NULL);
  g_assert (end >= begin);

  text = self->adapter.copy_text (instance, begin, end - begin + 1);
  bytes = g_bytes_new_take (text, strlen (text));

  spelling_job_add_fragment (job, bytes, begin, end - begin + 1);
}

static void
spelling_engine_add_fragments (SpellingEngine *self,
                               GObject        *instance,
                               SpellingJob    *job,
                               GtkBitset      *bitset)
{
  GtkBitsetIter iter;
  guint pos;

  g_assert (SPELLING_IS_ENGINE (self));
  g_assert (SPELLING_IS_JOB (job));
  g_assert (bitset != NULL);

  if (gtk_bitset_iter_init_first (&iter, bitset, &pos))
    {
      guint begin = pos;
      guint end = pos;

      while (gtk_bitset_iter_next (&iter, &pos))
        {
          if (pos != end + 1)
            {
              spelling_engine_add_fragment (self, instance, job, bitset, begin, end);
              begin = end = pos;
            }
          else
            {
              end = pos;
            }
        }

      spelling_engine_add_fragment (self, instance, job, bitset, begin, end);
    }
}

static gsize
spelling_engine_add_range (SpellingEngine *self,
                           GObject        *instance,
                           guint           begin,
                           guint           end,
                           GtkBitset      *all,
                           GtkBitset      *bitset)
{
  gsize ret;

  g_assert (SPELLING_IS_ENGINE (self));
  g_assert (self->active != NULL);
  g_assert (SPELLING_IS_JOB (self->active));
  g_assert (begin <= end);
  g_assert (all != NULL);
  g_assert (bitset != NULL);

  /* Track this range in "all" as we'll need to clear the areas
   * that have "no-spell-check" in our textregion too. We can
   * figure that out by subtracting bitset from all.
   */
  gtk_bitset_add_range (all, begin, end - begin);

  /* Track what the adapter thinks should be in this run */
  gtk_bitset_add_range (bitset, begin, end - begin);
  self->adapter.intersect_spellcheck_region (instance, bitset);

  /* And now subtract that from the all to cover the gaps */
  gtk_bitset_subtract (all, bitset);

  /* Add fragments for the sub-regions we need to check */
  spelling_engine_add_fragments (self, instance, self->active, bitset);

  /* Track the size so we can bail after sufficent data to check */
  ret = gtk_bitset_get_size (bitset);

  /* Reset bitset for next run */
  gtk_bitset_remove_all (bitset);

  return ret;
}

static gboolean
collect_ranges (gsize                   offset,
                const CjhTextRegionRun *run,
                gpointer                user_data)
{
  CollectRanges *collect = user_data;
  guint begin;
  guint end;

  if (run->data != TAG_NEEDS_CHECK)
    return FALSE;

  begin = offset;
  end = offset + run->length;

  spelling_engine_extend_range (collect->self, &begin, &end);

  collect->size += spelling_engine_add_range (collect->self,
                                              collect->instance,
                                              begin, end,
                                              collect->all,
                                              collect->bitset);

  return collect->size >= WATERMARK_PER_JOB;
}

static void
spelling_engine_job_finished (GObject      *object,
                              GAsyncResult *result,
                              gpointer      user_data)
{
  SpellingJob *job = (SpellingJob *)object;
  g_autoptr(GObject) instance = NULL;
  g_autoptr(SpellingEngine) self = user_data;
  g_autofree SpellingBoundary *fragments = NULL;
  g_autofree SpellingMistake *mistakes = NULL;
  guint n_fragments = 0;
  guint n_mistakes = 0;

  g_assert (SPELLING_IS_JOB (job));
  g_assert (G_IS_ASYNC_RESULT (result));
  g_assert (SPELLING_IS_ENGINE (self));

  g_clear_object (&self->active);

  if (!(instance = g_weak_ref_get (&self->instance_wr)))
    return;

  if (!self->adapter.check_enabled (instance))
    return;

  spelling_job_run_finish (job, result, &fragments, &n_fragments, &mistakes, &n_mistakes);

  for (guint f = 0; f < n_fragments; f++)
    {
      self->adapter.clear_tag (instance, fragments[f].offset, fragments[f].length);
      _cjh_text_region_replace (self->region,
                                fragments[f].offset, fragments[f].length,
                                TAG_CHECKED);
    }

  for (guint m = 0; m < n_mistakes; m++)
    self->adapter.apply_tag (instance, mistakes[m].offset, mistakes[m].length);

  /* Check immediately if there is more */
  if (spelling_engine_has_unchecked_regions (self))
    spelling_engine_queue_update (self, 0);
}

static void
spelling_engine_clear_runs (SpellingEngine *self,
                            GtkBitset      *bitset)
{
  GtkBitsetIter iter;
  guint pos;

  g_assert (SPELLING_IS_ENGINE (self));
  g_assert (bitset != NULL);

  if (gtk_bitset_iter_init_first (&iter, bitset, &pos))
    {
      guint begin = pos;
      guint end = pos;

      while (gtk_bitset_iter_next (&iter, &pos))
        {
          if (pos == end + 1)
            {
              end++;
              continue;
            }

          _cjh_text_region_replace (self->region, begin, end - begin + 1, TAG_CHECKED);

          begin = pos;
          end = pos;
        }

      _cjh_text_region_replace (self->region, begin, end - begin + 1, TAG_CHECKED);
    }
}

static gboolean
spelling_engine_tick (gpointer data)
{
  SpellingEngine *self = data;
  g_autoptr(GtkBitset) bitset = NULL;
  g_autoptr(GtkBitset) all = NULL;
  g_autoptr(GObject) instance = NULL;
  const CjhTextRegionRun *run;
  SpellingDictionary *dictionary;
  PangoLanguage *language;
  CollectRanges collect;
  gsize real_offset;
  guint cursor;

  g_assert (SPELLING_IS_ENGINE (self));
  g_assert (self->active == NULL);

  /* Be safe against weak-pointer lost or bad dictionary installations */
  if (!(instance = g_weak_ref_get (&self->instance_wr)) ||
      !(dictionary = self->adapter.get_dictionary (instance)) ||
      !(language = self->adapter.get_language (instance)))
    {
      g_clear_handle_id (&self->queued_update_handler, g_source_remove);
      return G_SOURCE_REMOVE;
    }

  self->active = spelling_job_new (dictionary, language);

  bitset = gtk_bitset_new_empty ();
  all = gtk_bitset_new_empty ();

  /* Always check the cursor location so that spellcheck feels snappy */
  cursor = self->adapter.get_cursor (instance);
  run = _cjh_text_region_get_run_at_offset (self->region, cursor, &real_offset);

  if (run == NULL || run->data == TAG_NEEDS_CHECK)
    {
      guint begin = cursor;
      guint end = cursor;

      if (spelling_engine_extend_range (self, &begin, &end))
        spelling_engine_add_range (self, instance, begin, end, all, bitset);
    }

  collect.self = self;
  collect.bitset = bitset;
  collect.all = all;
  collect.size = 0;
  collect.instance = instance;

  _cjh_text_region_foreach (self->region, collect_ranges, &collect);

  /* We need to clear everything from our textregion that is still
   * in @all as those are gaps in what should be checked, such as
   * no-spell-check regions.
   */
  spelling_engine_clear_runs (self, all);

  spelling_job_run (self->active,
                    spelling_engine_job_finished,
                    g_object_ref (self));

  g_clear_handle_id (&self->queued_update_handler, g_source_remove);

  return G_SOURCE_REMOVE;
}

static void
spelling_engine_queue_update (SpellingEngine *self,
                              guint           delay_msec)
{
  g_assert (SPELLING_IS_ENGINE (self));

  if (self->active != NULL)
    return;

  if (!spelling_engine_check_enabled (self))
    return;

  if (self->queued_update_handler == 0)
    self->queued_update_handler = g_timeout_add_full (G_PRIORITY_LOW,
                                                      delay_msec,
                                                      spelling_engine_tick,
                                                      self,
                                                      NULL);
}

static gboolean
spelling_engine_join_range (gsize                   offset,
                            const CjhTextRegionRun *left,
                            const CjhTextRegionRun *right)
{
  return left->data == right->data;
}

static void
spelling_engine_split_range (gsize                   offset,
                             const CjhTextRegionRun *run,
                             CjhTextRegionRun       *left,
                             CjhTextRegionRun       *right)
{
  /* We could potentially scan forward/back here from the offset
   * where the split occurs to mark the sub-regions as needing
   * to be checked. However we do that at another layer currently
   * so no need to do it here.
   *
   * It's also a bit difficult since you can't split the edge
   * runs into more than two runs and that would be necessary
   * if they were more than a couple words.
   */
}

static void
spelling_engine_dispose (GObject *object)
{
  SpellingEngine *self = (SpellingEngine *)object;

  g_clear_object (&self->active);
  g_clear_handle_id (&self->queued_update_handler, g_source_remove);
  g_weak_ref_set (&self->instance_wr, NULL);

  G_OBJECT_CLASS (spelling_engine_parent_class)->dispose (object);
}

static void
spelling_engine_finalize (GObject *object)
{
  SpellingEngine *self = (SpellingEngine *)object;

  g_weak_ref_clear (&self->instance_wr);
  g_clear_pointer (&self->region, _cjh_text_region_free);

  G_OBJECT_CLASS (spelling_engine_parent_class)->finalize (object);
}

static void
spelling_engine_class_init (SpellingEngineClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = spelling_engine_dispose;
  object_class->finalize = spelling_engine_finalize;
}

static void
spelling_engine_init (SpellingEngine *self)
{
  g_weak_ref_init (&self->instance_wr, NULL);

  self->region = _cjh_text_region_new (spelling_engine_join_range,
                                       spelling_engine_split_range);
}

SpellingEngine *
spelling_engine_new (const SpellingAdapter *adapter,
                     GObject               *instance)
{
  SpellingEngine *self;

  g_return_val_if_fail (adapter != NULL, NULL);
  g_return_val_if_fail (G_IS_OBJECT (instance), NULL);

  self = g_object_new (SPELLING_TYPE_ENGINE, NULL);
  g_weak_ref_set (&self->instance_wr, instance);
  self->adapter = *adapter;

  return self;
}

void
spelling_engine_before_insert_text (SpellingEngine *self,
                                    guint           position,
                                    guint           length)
{
  g_return_if_fail (SPELLING_IS_ENGINE (self));

  if (length == 0)
    return;

  if (self->active)
    spelling_job_notify_insert (self->active, position, length);

  _cjh_text_region_insert (self->region, position, length, TAG_NEEDS_CHECK);
}

void
spelling_engine_after_insert_text (SpellingEngine *self,
                                   guint           position,
                                   guint           length)
{
  g_return_if_fail (SPELLING_IS_ENGINE (self));

  if (length == 0)
    return;

  spelling_engine_invalidate (self, position, length);
}

void
spelling_engine_before_delete_range (SpellingEngine *self,
                                     guint           position,
                                     guint           length)
{
  g_return_if_fail (SPELLING_IS_ENGINE (self));

  if (length == 0)
    return;

  if (self->active)
    spelling_job_notify_delete (self->active, position, length);

  _cjh_text_region_remove (self->region, position, length);
}

void
spelling_engine_after_delete_range (SpellingEngine *self,
                                    guint           position)
{
  g_return_if_fail (SPELLING_IS_ENGINE (self));

  spelling_engine_invalidate (self, position, 0);
}

void
spelling_engine_iteration (SpellingEngine *self)
{
  g_return_if_fail (SPELLING_IS_ENGINE (self));

  if (self->active == NULL)
    spelling_engine_tick (self);
}

void
spelling_engine_invalidate_all (SpellingEngine *self)
{
  g_autoptr(GObject) instance = NULL;
  guint length;

  g_return_if_fail (SPELLING_IS_ENGINE (self));

  g_clear_object (&self->active);
  g_clear_handle_id (&self->queued_update_handler, g_source_remove);

  length = _cjh_text_region_get_length (self->region);

  if (length > 0)
    {
      _cjh_text_region_replace (self->region, 0, length, TAG_NEEDS_CHECK);

      if ((instance = g_weak_ref_get (&self->instance_wr)))
        self->adapter.clear_tag (instance, 0, length);
    }

  spelling_engine_queue_update (self, 0);
}

void
spelling_engine_invalidate (SpellingEngine *self,
                            guint           position,
                            guint           length)
{
  g_autoptr(GObject) instance = NULL;

  g_assert (SPELLING_IS_ENGINE (self));

  if (self->active)
    spelling_job_invalidate (self->active, position, length);

  _cjh_text_region_replace (self->region, position, length, TAG_NEEDS_CHECK);

  if ((instance = g_weak_ref_get (&self->instance_wr)))
    self->adapter.clear_tag (instance, position, length);

  spelling_engine_queue_update (self, 0);
}
