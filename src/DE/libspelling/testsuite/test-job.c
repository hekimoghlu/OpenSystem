/* test-job.c
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

#include <locale.h>

#include <glib/gi18n.h>

#include <libspelling.h>

#include "spelling-dictionary-internal.h"
#include "spelling-provider-internal.h"
#include "spelling-job-private.h"

static const char *test_dictionary[] = {
  "Straße",
  "a",
  "calle",
  "caminar",
  "die",
  "entlang",
  "gehen",
  "has",
  "it",
  "it's",
  "la",
  "misspelled",
  "not",
  "por",
  "say'",
  "text",
  "this",
  "word",
  "गच्छन्तु",
  "वीथिं",
  "དུ",
  "འགྲོ",
  "ལམ",
  "སྲང",
  NULL
};

#define TEST_TYPE_DICTIONARY (test_dictionary_get_type())

G_DECLARE_FINAL_TYPE (TestDictionary, test_dictionary, TEST, DICTIONARY, SpellingDictionary)

struct _TestDictionary
{
  SpellingDictionary parent_instance;
};

G_DEFINE_FINAL_TYPE (TestDictionary, test_dictionary, SPELLING_TYPE_DICTIONARY)

static gboolean
test_dictionary_contains_word (SpellingDictionary *dictionary,
                               const char         *word,
                               gssize              word_len)
{
  g_autofree char *copy = g_strndup (word, word_len < 0 ? strlen (word) : word_len);

  return g_strv_contains ((const char * const *)test_dictionary, copy);
}

static const char *
test_dictionary_get_extra_word_chars (SpellingDictionary *dictionary)
{
  return "'";
}

static void
test_dictionary_class_init (TestDictionaryClass *klass)
{
  SpellingDictionaryClass *dictionary_class = SPELLING_DICTIONARY_CLASS (klass);

  dictionary_class->contains_word = test_dictionary_contains_word;
  dictionary_class->get_extra_word_chars = test_dictionary_get_extra_word_chars;
}

static void
test_dictionary_init (TestDictionary *self)
{
}

#define TEST_TYPE_PROVIDER (test_provider_get_type())

G_DECLARE_FINAL_TYPE (TestProvider, test_provider, TEST, PROVIDER, SpellingProvider)

struct _TestProvider
{
  SpellingProvider  parent_instance;
  TestDictionary   *dictionary;
};

G_DEFINE_FINAL_TYPE (TestProvider, test_provider, SPELLING_TYPE_PROVIDER)

static SpellingDictionary *
test_provider_load_dictionary (SpellingProvider *provider,
                               const char       *language)
{
  TestProvider *self = TEST_PROVIDER (provider);

  return g_object_ref (SPELLING_DICTIONARY (self->dictionary));
}

static const char *
test_provider_get_default_code (SpellingProvider *provider)
{
  return "C";
}

static void
test_provider_finalize (GObject *object)
{
  TestProvider *self = (TestProvider *)object;

  g_clear_object (&self->dictionary);

  G_OBJECT_CLASS (test_provider_parent_class)->finalize (object);
}

static void
test_provider_class_init (TestProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  SpellingProviderClass *provider_class = SPELLING_PROVIDER_CLASS (klass);

  object_class->finalize = test_provider_finalize;

  provider_class->load_dictionary = test_provider_load_dictionary;
  provider_class->get_default_code = test_provider_get_default_code;
}

static void
test_provider_init (TestProvider *self)
{
  self->dictionary = g_object_new (TEST_TYPE_DICTIONARY, NULL);
}

static GMainLoop *main_loop;

typedef struct _TestJob
{
  const char *text;
  guint n_mistakes;
  const SpellingBoundary *mistakes;
} TestJob;

static void
validate (SpellingMistake *mistakes,
          guint            n_mistakes,
          TestJob         *test)
{
  g_assert_cmpint (n_mistakes, ==, test->n_mistakes);
  g_assert_true (mistakes != NULL || test->n_mistakes == 0);

  for (guint i = 0; i < n_mistakes; i++)
    {
      g_assert_cmpint (test->mistakes[i].offset, ==, mistakes[i].offset);
      g_assert_cmpint (test->mistakes[i].length, ==, mistakes[i].length);
    }
}

static void
job_finished (GObject      *object,
              GAsyncResult *result,
              gpointer      user_data)
{
  g_autofree SpellingMistake *mistakes = NULL;
  g_autofree SpellingBoundary *fragments = NULL;
  TestJob *test = user_data;
  GError *error = NULL;
  guint n_mistakes;
  guint n_fragments;

  g_assert (SPELLING_IS_JOB (object));

  spelling_job_run_finish (SPELLING_JOB (object), result, &fragments, &n_fragments, &mistakes, &n_mistakes);

#if 0
  for (guint i = 0; i < n_mistakes; i++)
    {
      g_autofree char *word = g_strndup (&test->text[mistakes[i].byte_offset], mistakes[i].byte_length);
      g_print ("Misspelled: \"%s\"\n", word);
    }
#endif

  g_assert_no_error (error);

  validate (mistakes, n_mistakes, test);

  g_main_loop_quit (main_loop);
}

static void
test_job_basic (void)
{
  g_autoptr(SpellingProvider) provider = g_object_new (TEST_TYPE_PROVIDER, NULL);
  const char *default_code = spelling_provider_get_default_code (provider);
  g_autoptr(SpellingDictionary) dictionary = spelling_provider_load_dictionary (provider, default_code);

  TestJob tests[] = {
    { "it's not a misspelled word", 0, NULL },
    { "not", 0, NULL },
    { "it's", 0, NULL },
    { "it's ", 0, NULL },
    { "\t\nsay' \t\n", 0, NULL },
    { "say'", 0, NULL },
    { "die Straße entlang gehen", 0, NULL },
    { "वीथिं गच्छन्तु", 0, NULL },
    { "སྲང་ལམ་དུ་འགྲོ།", 0, NULL },
    { "caminar por la   calle", 0, NULL },
    { "it'",
      1,
      (const SpellingBoundary[]) {
        { .offset = 0, .length = 3 },
      },
    },
    { "it' ",
      1,
      (const SpellingBoundary[]) {
        { .offset = 0, .length = 3 },
      },
    },
    { "this text has a misplled word",
      1,
      (const SpellingBoundary[]) {
        { .offset = 16, .length = 8 },
      },
    },
  };

  for (guint i = 0; i < G_N_ELEMENTS (tests); i++)
    {
      g_autoptr(GBytes) bytes = g_bytes_new (tests[i].text, strlen (tests[i].text));
      g_autoptr(SpellingJob) job = spelling_job_new (dictionary, pango_language_get_default ());
      g_autofree SpellingMistake *mistakes = NULL;
      g_autofree SpellingBoundary *fragments = NULL;
      guint n_mistakes;
      guint n_fragments;

      spelling_job_add_fragment (job, bytes, 0, g_utf8_strlen (tests[i].text, -1));

      /* Test async version */
      spelling_job_run (job, job_finished, &tests[i]);
      g_main_loop_run (main_loop);

      /* Now test sync version */
      spelling_job_run_sync (job, &fragments, &n_fragments, &mistakes, &n_mistakes);
      validate (mistakes, n_mistakes, &tests[i]);
    }
}

static void
test_job_discard (void)
{
  g_autoptr(SpellingProvider) provider = g_object_new (TEST_TYPE_PROVIDER, NULL);
  const char *default_code = spelling_provider_get_default_code (provider);
  g_autoptr(SpellingDictionary) dictionary = spelling_provider_load_dictionary (provider, default_code);
  g_autoptr(GBytes) bytes = g_bytes_new ("misplled word", 13);
  g_autoptr(SpellingJob) job = NULL;
  g_autofree SpellingMistake *mistakes = NULL;
  guint n_mistakes = 0;

  /* First make sure things work */
  job = spelling_job_new (dictionary, pango_language_get_default ());
  spelling_job_add_fragment (job, bytes, 1, 13);
  spelling_job_run_sync (job, NULL, NULL, &mistakes, &n_mistakes);
  g_assert_cmpint (n_mistakes, ==, 1);
  g_assert_cmpint (mistakes[0].offset, ==, 1);
  g_assert_cmpint (mistakes[0].length, ==, 8);
  g_clear_pointer (&mistakes, g_free);
  g_clear_object (&job);

  /* Now try to do an INSERT that SHOULD NOT collide */
  job = spelling_job_new (dictionary, pango_language_get_default ());
  spelling_job_add_fragment (job, bytes, 1, 13);
  spelling_job_notify_insert (job, 0, 3);
  spelling_job_run_sync (job, NULL, NULL, &mistakes, &n_mistakes);
  g_assert_cmpint (n_mistakes, ==, 1);
  g_assert_cmpint (mistakes[0].offset, ==, 1 + 3);
  g_assert_cmpint (mistakes[0].length, ==, 8);
  g_clear_pointer (&mistakes, g_free);
  g_clear_object (&job);

  /* Now try to do an INSERT that SHOULD collide */
  job = spelling_job_new (dictionary, pango_language_get_default ());
  spelling_job_add_fragment (job, bytes, 1, 13);
  spelling_job_notify_insert (job, 1, 3);
  spelling_job_run_sync (job, NULL, NULL, &mistakes, &n_mistakes);
  g_assert_cmpint (n_mistakes, ==, 0);
  g_assert_null (mistakes);
  g_clear_pointer (&mistakes, g_free);
  g_clear_object (&job);

  /* Now try to do a DELETE that SHOULD NOT collide */
  job = spelling_job_new (dictionary, pango_language_get_default ());
  spelling_job_add_fragment (job, bytes, 2, 13);
  spelling_job_notify_delete (job, 0, 1);
  spelling_job_run_sync (job, NULL, NULL, &mistakes, &n_mistakes);
  g_assert_cmpint (n_mistakes, ==, 1);
  g_assert_cmpint (mistakes[0].offset, ==, 1);
  g_assert_cmpint (mistakes[0].length, ==, 8);
  g_clear_pointer (&mistakes, g_free);
  g_clear_object (&job);

  /* Now try to do a DELETE that SHOULD collide */
  job = spelling_job_new (dictionary, pango_language_get_default ());
  spelling_job_add_fragment (job, bytes, 1, 13);
  spelling_job_notify_delete (job, 0, 1);
  spelling_job_run_sync (job, NULL, NULL, &mistakes, &n_mistakes);
  g_assert_cmpint (n_mistakes, ==, 0);
  g_assert_null (mistakes);
  g_clear_pointer (&mistakes, g_free);
  g_clear_object (&job);

  /* Now try to do a DELETE that SHOULD collide */
  job = spelling_job_new (dictionary, pango_language_get_default ());
  spelling_job_add_fragment (job, bytes, 0, 13);
  spelling_job_notify_delete (job, 13, 1);
  spelling_job_run_sync (job, NULL, NULL, &mistakes, &n_mistakes);
  g_assert_cmpint (n_mistakes, ==, 0);
  g_assert_null (mistakes);
  g_clear_pointer (&mistakes, g_free);
  g_clear_object (&job);

  /* Now try to do a DELETE that SHOULD NOT collide */
  job = spelling_job_new (dictionary, pango_language_get_default ());
  spelling_job_add_fragment (job, bytes, 0, 13);
  spelling_job_notify_delete (job, 14, 1);
  spelling_job_run_sync (job, NULL, NULL, &mistakes, &n_mistakes);
  g_assert_cmpint (n_mistakes, ==, 1);
  g_assert_cmpint (mistakes[0].offset, ==, 0);
  g_assert_cmpint (mistakes[0].length, ==, 8);
  g_clear_pointer (&mistakes, g_free);
  g_clear_object (&job);
}

int
main (int argc,
      char *argv[])
{
  main_loop = g_main_loop_new (NULL, FALSE);

  setlocale (LC_ALL, "");
  bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
  textdomain (GETTEXT_PACKAGE);

  g_test_init (&argc, &argv, NULL);
  g_test_add_func ("/Spelling/Job/basic", test_job_basic);
  g_test_add_func ("/Spelling/Job/discard", test_job_discard);
  return g_test_run ();
}
