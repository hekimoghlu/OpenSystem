/* test-json.c
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

#include <foundry.h>

static char *
serialize (JsonNode *node)
{
  g_autoptr(JsonGenerator) generator = json_generator_new ();
  json_generator_set_root (generator, node);
  return json_generator_to_data (generator, NULL);
}

static void
compare_json (const char *filename,
              JsonNode   *node)
{
  g_autofree char *path = NULL;
  g_autofree char *contents = NULL;
  g_autofree char *serialized = NULL;
  g_autoptr(GError) error = NULL;
  gsize len;

  g_assert_nonnull (g_getenv ("G_TEST_SRCDIR"));
  g_assert_nonnull (filename);
  g_assert_nonnull (node);

  path = g_build_filename (g_getenv ("G_TEST_SRCDIR"), "test-json", filename, NULL);

  g_file_get_contents (path, &contents, &len, &error);
  g_assert_no_error (error);
  g_assert_nonnull (contents);

  serialized = serialize (node);
  g_assert_nonnull (serialized);

  g_strstrip (contents);
  g_strstrip (serialized);

  g_assert_cmpstr (contents, ==, serialized);

  json_node_unref (node);
}

static void
test_json_object_new (void)
{
  JsonNode *temp = FOUNDRY_JSON_OBJECT_NEW ("key", "val");

  compare_json ("test1.json", FOUNDRY_JSON_OBJECT_NEW ("a", "b"));
  compare_json ("test1.json", FOUNDRY_JSON_OBJECT_NEW ("a", FOUNDRY_JSON_NODE_PUT_STRING ("b")));
  compare_json ("test2.json", FOUNDRY_JSON_OBJECT_NEW ("a", FOUNDRY_JSON_NODE_PUT_BOOLEAN (TRUE)));
  compare_json ("test3.json", FOUNDRY_JSON_OBJECT_NEW ("a", FOUNDRY_JSON_NODE_PUT_BOOLEAN (FALSE)));
  compare_json ("test4.json", FOUNDRY_JSON_OBJECT_NEW ("a", FOUNDRY_JSON_NODE_PUT_DOUBLE (123.45)));
  compare_json ("test5.json", FOUNDRY_JSON_OBJECT_NEW ("a", FOUNDRY_JSON_NODE_PUT_STRV (FOUNDRY_STRV_INIT ("a", "b", "c"))));
  compare_json ("test6.json", FOUNDRY_JSON_OBJECT_NEW ("a", FOUNDRY_JSON_NODE_PUT_INT (G_MAXINT)));
  compare_json ("test7.json",
                FOUNDRY_JSON_OBJECT_NEW ("a", "{",
                                           "aa", "bb",
                                         "}"));
  compare_json ("test8.json",
                FOUNDRY_JSON_OBJECT_NEW ("a", "[",
                                           "aa", "bb",
                                         "]"));
  compare_json ("test9.json", FOUNDRY_JSON_ARRAY_NEW ("a", "b", "c"));
  compare_json ("test10.json", FOUNDRY_JSON_ARRAY_NEW (FOUNDRY_JSON_NODE_PUT_NODE (NULL)));
  compare_json ("test11.json", FOUNDRY_JSON_ARRAY_NEW (FOUNDRY_JSON_NODE_PUT_NODE (temp)));

  json_node_unref (temp);
}

static JsonNode *
load_json (const char *name)
{
  g_autofree char *path = g_build_filename (g_getenv ("G_TEST_SRCDIR"), "test-json", name, NULL);
  g_autofree char *contents = NULL;
  g_autoptr(JsonParser) parser = json_parser_new ();
  g_autoptr(GError) error = NULL;
  gsize len;

  g_file_get_contents (path, &contents, &len, &error);
  g_assert_no_error (error);

  json_parser_load_from_data (parser, contents, len, &error);
  g_assert_no_error (error);

  return json_node_ref (json_parser_get_root (parser));
}

static void
test_json_node_parse (void)
{
  g_autoptr(JsonNode) test1 = load_json ("test1.json");
  g_autoptr(JsonNode) test2 = load_json ("test2.json");
  g_autoptr(JsonNode) test3 = load_json ("test3.json");
  g_autoptr(JsonNode) test4 = load_json ("test4.json");
  g_autoptr(JsonNode) test5 = load_json ("test5.json");
  g_autoptr(JsonNode) test6 = load_json ("test6.json");
  g_autoptr(JsonNode) test7 = load_json ("test7.json");
  g_autoptr(JsonNode) test8 = load_json ("test8.json");
  g_autoptr(JsonNode) test9 = load_json ("test9.json");
  const char *v_str;
  gboolean v_bool;
  double v_dbl;
  const char *v_str1;
  const char *v_str2;
  const char *v_str3;
  gint64 v_int;

  g_assert_true (FOUNDRY_JSON_OBJECT_PARSE (test1, "a", "b"));
  g_assert_false (FOUNDRY_JSON_OBJECT_PARSE (test1, "a", "b", "c", "d"));

  g_assert_true (FOUNDRY_JSON_OBJECT_PARSE (test2, "a", FOUNDRY_JSON_NODE_GET_BOOLEAN (&v_bool)));
  g_assert_false (FOUNDRY_JSON_OBJECT_PARSE (test2, "a", FOUNDRY_JSON_NODE_GET_STRING (&v_str)));
  g_assert_true (v_bool);

  g_assert_true (FOUNDRY_JSON_OBJECT_PARSE (test3, "a", FOUNDRY_JSON_NODE_GET_BOOLEAN (&v_bool)));
  g_assert_false (v_bool);

  g_assert_true (FOUNDRY_JSON_OBJECT_PARSE (test4, "a", FOUNDRY_JSON_NODE_GET_DOUBLE (&v_dbl)));
  g_assert_cmpfloat (v_dbl, ==, 123.45);

  g_assert_true (FOUNDRY_JSON_OBJECT_PARSE (test5,
                                            "a", "[",
                                              FOUNDRY_JSON_NODE_GET_STRING (&v_str1),
                                              FOUNDRY_JSON_NODE_GET_STRING (&v_str2),
                                              FOUNDRY_JSON_NODE_GET_STRING (&v_str3),
                                            "]"));
  g_assert_cmpstr (v_str1, ==, "a");
  g_assert_cmpstr (v_str2, ==, "b");
  g_assert_cmpstr (v_str3, ==, "c");

  g_assert_true (FOUNDRY_JSON_OBJECT_PARSE (test6, "a", FOUNDRY_JSON_NODE_GET_INT (&v_int)));
  g_assert_cmpint (2147483647, ==, v_int);

  g_assert_true (FOUNDRY_JSON_OBJECT_PARSE (test7, "a", "{", "aa", "bb", "}"));
  g_assert_true (FOUNDRY_JSON_OBJECT_PARSE (test8, "a", "[", "aa", "bb", "]"));

  g_assert_true (FOUNDRY_JSON_ARRAY_PARSE (test9, "a", "b", "c"));
}

int
main (int argc,
      char *argv[])
{
  g_test_init (&argc, &argv, NULL);
  g_test_add_func ("/Foundry/Json/Node/new", test_json_object_new);
  g_test_add_func ("/Foundry/Json/Node/parse", test_json_node_parse);
  return g_test_run ();
}
