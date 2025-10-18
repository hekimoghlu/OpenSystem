/* test-region.c
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

#include "cjhtextregionprivate.h"
#include "cjhtextregionbtree.h"

static void
get_run_at_offset (void)
{
  CjhTextRegion *region = _cjh_text_region_new (NULL, NULL);
  const CjhTextRegionRun *run;
  gsize offset;

  _cjh_text_region_insert (region, 0, 10, GUINT_TO_POINTER (1));
  _cjh_text_region_insert (region, 10, 10, GUINT_TO_POINTER (2));
  _cjh_text_region_replace (region, 15, 1, GUINT_TO_POINTER (3));

  run = _cjh_text_region_get_run_at_offset (region, 15, &offset);

  g_assert_true (run != NULL);
  g_assert_cmpint (offset, ==, 15);
  g_assert_cmpint (run->length, ==, 1);
  g_assert_true (run->data == GUINT_TO_POINTER (3));

  _cjh_text_region_free (region);
}

static void
assert_leaves_empty (CjhTextRegion *region)
{
  CjhTextRegionNode *leaf = _cjh_text_region_get_first_leaf (region);
  guint count = 0;

  for (; leaf; leaf = leaf->leaf.next, count++)
    {
      CjhTextRegionNode *parent = cjh_text_region_node_get_parent (leaf);
      guint length = cjh_text_region_node_length (leaf);
      guint length_in_parent = 0;

      SORTED_ARRAY_FOREACH (&parent->branch.children, CjhTextRegionChild, child, {
        if (child->node == leaf)
          {
            length_in_parent = child->length;
            break;
          }
      });

      if (length || length_in_parent)
        g_error ("leaf %p %u has length of %u in %u runs. Parent thinks it has length of %u.",
                 leaf, count, length, SORTED_ARRAY_LENGTH (&leaf->leaf.runs), length_in_parent);
    }
}

static guint
count_leaves (CjhTextRegion *region)
{
  CjhTextRegionNode *leaf = _cjh_text_region_get_first_leaf (region);
  guint count = 0;

  for (; leaf; leaf = leaf->leaf.next)
    count++;

  return count;
}

static guint
count_internal_recuse (CjhTextRegionNode *node)
{
  guint count = 1;

  g_assert (!cjh_text_region_node_is_leaf (node));

  SORTED_ARRAY_FOREACH (&node->branch.children, CjhTextRegionChild, child, {
    g_assert (child->node != NULL);

    if (!cjh_text_region_node_is_leaf (child->node))
      count += count_internal_recuse (child->node);
  });

  return count;
}

static guint
count_internal (CjhTextRegion *region)
{
  return count_internal_recuse (&region->root);
}

G_GNUC_UNUSED static inline void
print_tree (CjhTextRegionNode *node,
            guint              depth)
{
  for (guint i = 0; i < depth; i++)
    g_print ("  ");
  g_print ("%p %s Length=%"G_GSIZE_MODIFIER"u Items=%u Prev<%p> Next<%p>\n",
           node,
           cjh_text_region_node_is_leaf (node) ? "Leaf" : "Branch",
           cjh_text_region_node_length (node),
           cjh_text_region_node_is_leaf (node) ?
             SORTED_ARRAY_LENGTH (&node->leaf.runs) :
             SORTED_ARRAY_LENGTH (&node->branch.children),
           cjh_text_region_node_is_leaf (node) ? node->leaf.prev : node->branch.prev,
           cjh_text_region_node_is_leaf (node) ? node->leaf.next : node->branch.next);
  if (!cjh_text_region_node_is_leaf (node))
    {
      SORTED_ARRAY_FOREACH (&node->branch.children, CjhTextRegionChild, child, {
        print_tree (child->node, depth+1);
      });
    }
}

static void
assert_empty (CjhTextRegion *region)
{
#if 0
  print_tree (&region->root, 0);
#endif

  g_assert_cmpint (_cjh_text_region_get_length (region), ==, 0);
  assert_leaves_empty (region);
  g_assert_cmpint (1, ==, count_internal (region));
  g_assert_cmpint (1, ==, count_leaves (region));
}

static gboolean
non_overlapping_insert_remove_cb (gsize                   offset,
                                  const CjhTextRegionRun *run,
                                  gpointer                user_data)
{
  g_assert_cmpint (offset, ==, GPOINTER_TO_UINT (run->data));
  return FALSE;
}

static void
non_overlapping_insert_remove (void)
{
  CjhTextRegion *region = _cjh_text_region_new (NULL, NULL);

  assert_empty (region);

  for (guint i = 0; i < 100000; i++)
    {
      _cjh_text_region_insert (region, i, 1, GUINT_TO_POINTER (i));
      g_assert_cmpint (_cjh_text_region_get_length (region), ==, i + 1);
    }

  g_assert_cmpint (_cjh_text_region_get_length (region), ==, 100000);

  _cjh_text_region_foreach (region, non_overlapping_insert_remove_cb, NULL);

  for (guint i = 0; i < 100000; i++)
    _cjh_text_region_remove (region, 100000-1-i, 1);

  g_assert_cmpint (_cjh_text_region_get_length (region), ==, 0);

  assert_empty (region);

  _cjh_text_region_free (region);
}

typedef struct {
  gsize offset;
  gsize length;
  gpointer data;
} SplitRunCheck;

typedef struct {
  gsize index;
  gsize count;
  const SplitRunCheck *checks;
} SplitRun;

static gboolean
split_run_cb (gsize                   offset,
              const CjhTextRegionRun *run,
              gpointer                user_data)
{
  SplitRun *state = user_data;
  g_assert_cmpint (offset, ==, state->checks[state->index].offset);
  g_assert_cmpint (run->length, ==, state->checks[state->index].length);
  g_assert_true (run->data == state->checks[state->index].data);
  state->index++;
  return FALSE;
}

static void
split_run (void)
{
  static const SplitRunCheck checks[] = {
    { 0, 1, NULL },
    { 1, 1, GSIZE_TO_POINTER (1) },
    { 2, 1, NULL },
  };
  SplitRun state = { 0, 3, checks };
  CjhTextRegion *region = _cjh_text_region_new (NULL, NULL);
  _cjh_text_region_insert (region, 0, 2, NULL);
  g_assert_cmpint (_cjh_text_region_get_length (region), ==, 2);
  _cjh_text_region_insert (region, 1, 1, GSIZE_TO_POINTER (1));
  g_assert_cmpint (_cjh_text_region_get_length (region), ==, 3);
  _cjh_text_region_foreach (region, split_run_cb, &state);
  _cjh_text_region_free (region);
}

static gboolean
can_join_cb (gsize                   offset,
             const CjhTextRegionRun *left,
             const CjhTextRegionRun *right)
{
  return left->data == right->data;
}

static void
no_split_run (void)
{
  static const SplitRunCheck checks[] = {
    { 0, 3, NULL },
  };
  SplitRun state = { 0, 1, checks };
  CjhTextRegion *region = _cjh_text_region_new (can_join_cb, NULL);
  _cjh_text_region_insert (region, 0, 2, NULL);
  g_assert_cmpint (_cjh_text_region_get_length (region), ==, 2);
  _cjh_text_region_insert (region, 1, 1, NULL);
  g_assert_cmpint (_cjh_text_region_get_length (region), ==, 3);
  _cjh_text_region_foreach (region, split_run_cb, &state);
  _cjh_text_region_free (region);
}

static void
random_insertion (void)
{
  CjhTextRegion *region = _cjh_text_region_new (NULL, NULL);
  gsize expected = 0;

  for (guint i = 0; i < 10000; i++)
    {
      guint pos = g_random_int_range (0, region->length + 1);
      guint len = g_random_int_range (1, 20);

      _cjh_text_region_insert (region, pos, len, GUINT_TO_POINTER (i));

      expected += len;
    }

  g_assert_cmpint (expected, ==, region->length);

  _cjh_text_region_replace (region, 0, region->length, NULL);
  g_assert_cmpint (count_leaves (region), ==, 1);
  g_assert_cmpint (count_internal (region), ==, 1);

  _cjh_text_region_free (region);
}

static void
random_deletion (void)
{
  CjhTextRegion *region = _cjh_text_region_new (NULL, NULL);

  _cjh_text_region_insert (region, 0, 10000, NULL);

  while (region->length > 0)
    {
      guint pos = region->length > 1 ? g_random_int_range (0, region->length-1) : 0;
      guint len = region->length - pos > 1 ? g_random_int_range (1, region->length - pos) : 1;

      _cjh_text_region_remove (region, pos, len);
    }

  _cjh_text_region_free (region);
}

static void
random_insert_deletion (void)
{
  CjhTextRegion *region = _cjh_text_region_new (NULL, NULL);
  guint expected = 0;
  guint i = 0;

  while (region->length < 10000)
    {
      guint pos = g_random_int_range (0, region->length + 1);
      guint len = g_random_int_range (1, 20);

      _cjh_text_region_insert (region, pos, len, GUINT_TO_POINTER (i));

      expected += len;
      i++;
    }

  g_assert_cmpint (expected, ==, region->length);

  while (region->length > 0)
    {
      guint pos = region->length > 1 ? g_random_int_range (0, region->length-1) : 0;
      guint len = region->length - pos > 1 ? g_random_int_range (1, region->length - pos) : 1;

      g_assert (pos + len <= region->length);

      _cjh_text_region_remove (region, pos, len);
    }

  _cjh_text_region_free (region);
}

static void
test_val_queue (void)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"

  VAL_QUEUE_NODE(guint8, 32) field;
  guint8 pos;

  VAL_QUEUE_INIT (&field);

  for (guint i = 0; i < 32; i++)
    VAL_QUEUE_PUSH_TAIL (&field, i);
  g_assert_cmpint (VAL_QUEUE_LENGTH (&field), ==, 32);

  for (guint i = 0; i < 32; i++)
    {
      VAL_QUEUE_NTH (&field, i, pos);
      g_assert_cmpint (pos, ==, i);
    }
  for (guint i = 0; i < 32; i++)
    {
      VAL_QUEUE_POP_HEAD (&field, pos);
      g_assert_cmpint (pos, ==, i);
    }
  g_assert_cmpint (VAL_QUEUE_LENGTH (&field), ==, 0);

  for (guint i = 0; i < 32; i++)
    VAL_QUEUE_PUSH_TAIL (&field, i);
  g_assert_cmpint (VAL_QUEUE_LENGTH (&field), ==, 32);
  for (guint i = 0; i < 32; i++)
    {
      VAL_QUEUE_POP_TAIL (&field, pos);
      g_assert_cmpint (pos, ==, 31-i);
    }
  g_assert_cmpint (VAL_QUEUE_LENGTH (&field), ==, 0);

  for (guint i = 0; i < 32; i++)
    VAL_QUEUE_PUSH_TAIL (&field, i);
  while (VAL_QUEUE_LENGTH (&field))
    VAL_QUEUE_POP_NTH (&field, VAL_QUEUE_LENGTH (&field)/2, pos);

#pragma GCC diagnostic pop
}

typedef struct {
  int v;
} Dummy;

static void
sorted_array (void)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"

  SORTED_ARRAY_FIELD (Dummy, 32) field;
  Dummy d;
  guint i;

  SORTED_ARRAY_INIT (&field);

  d.v = 0; SORTED_ARRAY_INSERT_VAL (&field, 0, d);
  d.v = 2; SORTED_ARRAY_INSERT_VAL (&field, 1, d);
  d.v = 1; SORTED_ARRAY_INSERT_VAL (&field, 1, d);
  i = 0;
  g_assert_cmpint (SORTED_ARRAY_LENGTH (&field), ==, 3);
  SORTED_ARRAY_FOREACH (&field, Dummy, dummy, {
    g_assert_cmpint (dummy->v, ==, i++);
  });
  g_assert_cmpint (i, ==, 3);
  SORTED_ARRAY_POP_HEAD (&field, d); g_assert_cmpint (d.v, ==, 0);
  SORTED_ARRAY_POP_HEAD (&field, d); g_assert_cmpint (d.v, ==, 1);
  SORTED_ARRAY_POP_HEAD (&field, d); g_assert_cmpint (d.v, ==, 2);

  for (i = 0; i < 10; i++)
    { d.v = i * 2;
      SORTED_ARRAY_INSERT_VAL (&field, i, d); }
  for (i = 0; i < 10; i++)
    { d.v = i * 2 + 1;
      SORTED_ARRAY_INSERT_VAL (&field, i*2+1, d); }
  i = 0;
  g_assert_cmpint (SORTED_ARRAY_LENGTH (&field), ==, 20);
  SORTED_ARRAY_FOREACH (&field, Dummy, dummy, {
    g_assert_cmpint (dummy->v, ==, i++);
  });
  g_assert_cmpint (i, ==, 20);
  SORTED_ARRAY_FOREACH (&field, Dummy, dummy, {
    (void)dummy;
    SORTED_ARRAY_FOREACH_REMOVE (&field);
  });
  g_assert_cmpint (SORTED_ARRAY_LENGTH (&field), ==, 0);


  for (i = 0; i < 32; i++)
    {
      d.v = i;
      SORTED_ARRAY_PUSH_TAIL (&field, d);
    }
  g_assert_cmpint (32, ==, SORTED_ARRAY_LENGTH (&field));
  i = 0;
  SORTED_ARRAY_FOREACH (&field, Dummy, dummy, {
    g_assert_cmpint (dummy->v, ==, i);
    g_assert_cmpint (SORTED_ARRAY_LENGTH (&field), ==, 32-i);
    SORTED_ARRAY_FOREACH_REMOVE (&field);
    i++;
  });
  g_assert_cmpint (0, ==, SORTED_ARRAY_LENGTH (&field));

  for (i = 0; i < 32; i++)
    {
      d.v = i;
      SORTED_ARRAY_PUSH_TAIL (&field, d);
    }
  g_assert_cmpint (32, ==, SORTED_ARRAY_LENGTH (&field));
  i = 31;
  SORTED_ARRAY_FOREACH_REVERSE (&field, Dummy, dummy, {
    g_assert_cmpint (dummy->v, ==, i);
    SORTED_ARRAY_REMOVE_INDEX (&field, i, d);
    i--;
  });

#pragma GCC diagnostic pop
}

static gboolean
replace_part_of_long_run_join (gsize                   offset,
                               const CjhTextRegionRun *left,
                               const CjhTextRegionRun *right)
{
  return FALSE;
}

static void
replace_part_of_long_run_split (gsize                   offset,
                                const CjhTextRegionRun *run,
                                CjhTextRegionRun       *left,
                                CjhTextRegionRun       *right)
{
  left->data = run->data;
  right->data = GSIZE_TO_POINTER (GPOINTER_TO_SIZE (run->data) + left->length);
}

static void
replace_part_of_long_run (void)
{
  CjhTextRegion *region = _cjh_text_region_new (replace_part_of_long_run_join,
                                                replace_part_of_long_run_split);
  static const SplitRunCheck checks0[] = {
    { 0, 5, NULL },
  };
  static const SplitRunCheck checks1[] = {
    { 0, 1, NULL },
    { 1, 3, GSIZE_TO_POINTER (2) },
  };
  static const SplitRunCheck checks2[] = {
    { 0, 1, GSIZE_TO_POINTER (0) },
    { 1, 1, GSIZE_TO_POINTER ((1L<<31)|1) },
    { 2, 3, GSIZE_TO_POINTER (2) },
  };
  static const SplitRunCheck checks3[] = {
    { 0, 1, GSIZE_TO_POINTER (0) },
    { 1, 1, GSIZE_TO_POINTER ((1L<<31)|1) },
    { 2, 1, GSIZE_TO_POINTER (2) },
    { 3, 1, GSIZE_TO_POINTER (4) },
  };
  static const SplitRunCheck checks4[] = {
    { 0, 1, GSIZE_TO_POINTER (0) },
    { 1, 1, GSIZE_TO_POINTER ((1L<<31)|1) },
    { 2, 1, GSIZE_TO_POINTER (2) },
    { 3, 1, GSIZE_TO_POINTER ((1L<<31)|2) },
    { 4, 1, GSIZE_TO_POINTER (4) },
  };
  SplitRun state0 = { 0, 1, checks0 };
  SplitRun state1 = { 0, 2, checks1 };
  SplitRun state2 = { 0, 3, checks2 };
  SplitRun state3 = { 0, 4, checks3 };
  SplitRun state4 = { 0, 5, checks4 };

  _cjh_text_region_insert (region, 0, 5, NULL);
  _cjh_text_region_foreach (region, split_run_cb, &state0);
  _cjh_text_region_remove (region, 1, 1);
  _cjh_text_region_foreach (region, split_run_cb, &state1);
  _cjh_text_region_insert (region, 1, 1, GSIZE_TO_POINTER ((1L<<31)|1));
  _cjh_text_region_foreach (region, split_run_cb, &state2);
  _cjh_text_region_remove (region, 3, 1);
  _cjh_text_region_foreach (region, split_run_cb, &state3);
  _cjh_text_region_insert (region, 3, 1, GSIZE_TO_POINTER ((1L<<31)|2));
  _cjh_text_region_foreach (region, split_run_cb, &state4);
  _cjh_text_region_free (region);
}

typedef struct
{
  char *original;
  char *changes;
  GString *res;
} wordstate;

static gboolean
word_foreach_cb (gsize                   offset,
                 const CjhTextRegionRun *run,
                 gpointer                data)
{
  wordstate *state = data;
  gsize sdata = GPOINTER_TO_SIZE (run->data);
  gsize soff = sdata & ~(1L<<31);
  char *src;

  if (sdata == soff)
    src = state->original;
  else
    src = state->changes;

#if 0
  g_print ("%lu len %lu (%s at %lu) %s\n",
           offset, run->length, sdata == soff ? "original" : "changes", soff,
           sdata == soff && src[sdata] == '\n' ? "is-newline" : "");
#endif

  g_string_append_len (state->res, src + soff, run->length);

  return FALSE;
}

static gboolean
join_word_cb (gsize                   offset,
              const CjhTextRegionRun *left,
              const CjhTextRegionRun *right)
{
  return FALSE;
}

static void
split_word_cb (gsize                   offset,
               const CjhTextRegionRun *run,
               CjhTextRegionRun       *left,
               CjhTextRegionRun       *right)
{
  gsize sdata = GPOINTER_TO_SIZE (run->data);

  left->data = run->data;
  right->data = GSIZE_TO_POINTER (sdata + left->length);
}

static void
test_words_database (void)
{
  CjhTextRegion *region;
  g_autofree char *contents = NULL;
  g_autoptr(GString) str = NULL;
  g_autoptr(GString) res = NULL;
  const char *word;
  const char *iter;
  gsize len;
  wordstate state;

  if (!g_file_get_contents ("/usr/share/dict/words", &contents, &len, NULL))
    {
      g_test_skip ("Words database not available");
      return;
    }

  region = _cjh_text_region_new (join_word_cb, split_word_cb);
  str = g_string_new (NULL);
  res = g_string_new (NULL);

  /* 0 offset of base buffer */
  _cjh_text_region_insert (region, 0, len, NULL);

  /* For each each word, remove it and replace it with a word added to str.
   * At the end we'll create the buffer and make sure we get the same.
   */
  word = contents;
  iter = contents;
  for (;;)
    {
      if (*iter == 0)
        break;

      if (g_unichar_isspace (g_utf8_get_char (iter)))
        {
          gsize pos = str->len;

          g_string_append_len (str, word, iter - word);

          _cjh_text_region_replace (region, word - contents, iter - word, GSIZE_TO_POINTER ((1L<<31)|pos));

          while (*iter && g_unichar_isspace (g_utf8_get_char (iter)))
            iter = g_utf8_next_char (iter);
          word = iter;
        }
      else
        iter = g_utf8_next_char (iter);
    }

  state.original = contents;
  state.changes = str->str;
  state.res = res;
  _cjh_text_region_foreach (region, word_foreach_cb, &state);
  _cjh_text_region_free (g_steal_pointer (&region));

  g_assert_true (g_str_equal (contents, res->str));
}

static gboolean
foreach_cb (gsize                   offset,
            const CjhTextRegionRun *run,
            gpointer                user_data)
{
  guint *count = user_data;

  g_assert_cmpint (GPOINTER_TO_SIZE (run->data), ==, offset);
  (*count)++;

  return FALSE;
}

static void
foreach_in_range (void)
{
  CjhTextRegion *region = _cjh_text_region_new (NULL, NULL);
  guint count;

  for (guint i = 0; i < 100000; i++)
    {
      _cjh_text_region_insert (region, i, 1, GUINT_TO_POINTER (i));
      g_assert_cmpint (_cjh_text_region_get_length (region), ==, i + 1);
    }

  count = 0;
  _cjh_text_region_foreach_in_range (region, 0, 100000, foreach_cb, &count);
  g_assert_cmpint (count, ==, 100000);

  count = 0;
  _cjh_text_region_foreach_in_range (region, 1000, 5000, foreach_cb, &count);
  g_assert_cmpint (count, ==, 4000);

  _cjh_text_region_replace (region, 0, 10000, NULL);

  count = 0;
  _cjh_text_region_foreach_in_range (region, 1000, 5000, foreach_cb, &count);
  g_assert_cmpint (count, ==, 1);

  _cjh_text_region_free (region);
}

static void
full_tail_node (void)
{
  CjhTextRegion *region = _cjh_text_region_new (NULL, NULL);

  for (guint i = 0; i < CJH_TEXT_REGION_MAX_RUNS-3; i++)
    _cjh_text_region_insert (region, i, 1, GUINT_TO_POINTER (i));
  _cjh_text_region_insert (region, CJH_TEXT_REGION_MAX_RUNS-3, 100, GUINT_TO_POINTER (1000));

  _cjh_text_region_remove (region, CJH_TEXT_REGION_MAX_RUNS-1, 1);
  _cjh_text_region_remove (region, CJH_TEXT_REGION_MAX_RUNS, 1);
  _cjh_text_region_remove (region, CJH_TEXT_REGION_MAX_RUNS+1, 1);

  _cjh_text_region_free (region);
}

int
main (int argc,
      char *argv[])
{
  g_test_init (&argc, &argv, NULL);
  g_test_add_func ("/Cjh/TextRegion/val_queue", test_val_queue);
  g_test_add_func ("/Cjh/TextRegion/sorted_array", sorted_array);
  g_test_add_func ("/Cjh/TextRegion/non_overlapping_insert_remove", non_overlapping_insert_remove);
  g_test_add_func ("/Cjh/TextRegion/foreach_in_range", foreach_in_range);
  g_test_add_func ("/Cjh/TextRegion/split_run", split_run);
  g_test_add_func ("/Cjh/TextRegion/no_split_run", no_split_run);
  g_test_add_func ("/Cjh/TextRegion/random_insertion", random_insertion);
  g_test_add_func ("/Cjh/TextRegion/random_deletion", random_deletion);
  g_test_add_func ("/Cjh/TextRegion/random_insert_deletion", random_insert_deletion);
  g_test_add_func ("/Cjh/TextRegion/replace_part_of_long_run", replace_part_of_long_run);
  g_test_add_func ("/Cjh/TextRegion/words_database", test_words_database);
  g_test_add_func ("/Cjh/TextRegion/get_run_at_offset", get_run_at_offset);
  g_test_add_func ("/Cjh/TextRegion/full_tail_node", full_tail_node);
  return g_test_run ();
}
