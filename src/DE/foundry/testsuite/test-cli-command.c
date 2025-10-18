/* test-cli-command.c
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

#include <foundry.h>

#include "foundry-cli-command-tree-private.h"
#include "foundry-command-line-local-private.h"
#include "foundry-util-private.h"

static FoundryCommandLine *command_line;

static int
test_command (FoundryCommandLine *_command_line,
              const char * const *argv,
              FoundryCliOptions  *options,
              DexCancellable     *cancellable)
{
  return 0;
}

static void
test_tree1 (void)
{
  g_autoptr(FoundryCliCommandTree) tree = foundry_cli_command_tree_new ();
  g_autoptr(GError) error = NULL;

  const FoundryCliCommand command_def = {
    (GOptionEntry[]) {
      {"string", 's', 0, G_OPTION_ARG_STRING},
      {"int", 'i', 0, G_OPTION_ARG_INT},
      {0}
    },
    test_command,
  };

  const FoundryCliCommand *command;
  g_autoptr(FoundryCliOptions) options = NULL;
  g_auto(GStrv) args = NULL;
  int ival = 0;

  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "test", "command"),
                                     &command_def);

  args = g_strdupv ((char **)FOUNDRY_STRV_INIT ("foundry", "test", "command", "-i", "3", "option"));
  command = foundry_cli_command_tree_lookup (tree, &args, &options, &error);

  g_assert_no_error (error);
  g_assert_true (command->run == command_def.run);
  g_assert_nonnull (args);
  g_assert_cmpint (g_strv_length (args), ==, 2);
  g_assert_cmpstr ("foundry-test-command", ==, args[0]);
  g_assert_cmpstr ("option", ==, args[1]);

  g_assert_true (foundry_cli_options_get_int (options, "int", &ival));
  g_assert_cmpint (3, ==, ival);
  g_assert_null (foundry_cli_options_get_string (options, "string"));
}

static void
test_tree2 (void)
{
  g_autoptr(FoundryCliCommandTree) tree = foundry_cli_command_tree_new ();
  g_autoptr(FoundryCliOptions) options = NULL;
  const FoundryCliCommand *command;
  g_autoptr(GError) error = NULL;
  g_auto(GStrv) args = NULL;

  args = g_strdupv ((char **)FOUNDRY_STRV_INIT ("foundry"));
  command = foundry_cli_command_tree_lookup (tree, &args, &options, &error);

  g_assert_error (error, G_IO_ERROR, G_IO_ERROR_NOT_SUPPORTED);
  g_assert_null (command);
  g_assert_null (options);
}

static void
test_tree3 (void)
{
  g_autoptr(FoundryCliCommandTree) tree = foundry_cli_command_tree_new ();
  g_autoptr(GError) error = NULL;

  const FoundryCliCommand command_def = {
    NULL,
    test_command,
  };

  const FoundryCliCommand *command;
  g_autoptr(FoundryCliOptions) options = NULL;
  g_auto(GStrv) args = NULL;

  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "test"),
                                     &command_def);

  args = g_strdupv ((char **)FOUNDRY_STRV_INIT ("foundry", "missing"));
  command = foundry_cli_command_tree_lookup (tree, &args, &options, &error);

  g_assert_nonnull (args);
  g_assert_cmpstr (args[0], ==, "foundry");
  g_assert_cmpstr (args[1], ==, "missing");
  g_assert_null (args[2]);

  g_assert_null (command);
  g_assert_nonnull (options);
}

static void
test_complete1 (void)
{
  const FoundryCliCommand command_def = { NULL, test_command, };
  g_autoptr(FoundryCliCommandTree) tree = foundry_cli_command_tree_new ();
  g_autoptr(FoundryCliOptions) options = NULL;
  g_auto(GStrv) comp = NULL;

  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "test"),
                                     &command_def);

  comp = foundry_cli_command_tree_complete (tree, command_line, "foundry ", 8, "");

  g_assert_nonnull (comp);
  g_assert_cmpint (g_strv_length (comp), ==, 1);
  g_assert_cmpstr (comp[0], ==, "test ");
}

static void
test_complete2 (void)
{
  const FoundryCliCommand command_def = { NULL, test_command, };
  g_autoptr(FoundryCliCommandTree) tree = foundry_cli_command_tree_new ();
  g_autoptr(FoundryCliOptions) options = NULL;
  g_auto(GStrv) comp = NULL;

  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "test", "this"),
                                     &command_def);
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "test", "that"),
                                     &command_def);

  comp = foundry_cli_command_tree_complete (tree, command_line, "foundry test t", 14, "t");

  g_assert_nonnull (comp);
  g_assert_cmpint (g_strv_length (comp), ==, 2);
  g_assert_cmpstr (comp[0], ==, "this ");
  g_assert_cmpstr (comp[1], ==, "that ");
  g_assert_null (comp[2]);
}

static void
test_complete3 (void)
{
  const FoundryCliCommand command_def = {
    (GOptionEntry[]) {
      { "long", 'l', 0, G_OPTION_ARG_NONE },
      { 0 }
    },
    test_command,
  };

  g_autoptr(FoundryCliCommandTree) tree = foundry_cli_command_tree_new ();
  g_autoptr(FoundryCliOptions) options = NULL;
  g_auto(GStrv) comp = NULL;

  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "test", "this"),
                                     &command_def);
  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "test", "that"),
                                     &command_def);

  comp = foundry_cli_command_tree_complete (tree, command_line, "foundry test this -", 19, "-");

  g_assert_nonnull (comp);
  g_assert_cmpint (g_strv_length (comp), ==, 2);
  g_assert_cmpstr (comp[0], ==, "--long ");
  g_assert_cmpstr (comp[1], ==, "-l");
  g_assert_null (comp[2]);
}

static void
test_complete4 (void)
{
  const FoundryCliCommand command_def = {
    (GOptionEntry[]) {
      { "test", 0, 0, G_OPTION_ARG_NONE },
      { "file", 0, 0, G_OPTION_ARG_FILENAME },
      { 0 }
    },
    test_command,
  };

  g_autoptr(FoundryCliCommandTree) tree = foundry_cli_command_tree_new ();
  g_autoptr(FoundryCliOptions) options = NULL;
  g_auto(GStrv) comp = NULL;

  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "monitor"),
                                     &command_def);

  comp = foundry_cli_command_tree_complete (tree, command_line, "foundry monitor --test --file ", 30, "");

  g_assert_nonnull (comp);
  g_assert_cmpint (g_strv_length (comp), ==, 1);
  g_assert_cmpstr (comp[0], ==, "__FOUNDRY_FILE");
  g_assert_null (comp[1]);
}

static char **
test_complete5_complete (FoundryCommandLine *_command_line,
                         const char         *command,
                         const GOptionEntry *entry,
                         FoundryCliOptions  *options,
                         const char * const *argv,
                         const char         *current)
{
  if (entry != NULL)
    return g_strdupv ((char **)FOUNDRY_STRV_INIT ("first", "second", "third"));

  return NULL;
}

static void
test_complete5 (void)
{
  const FoundryCliCommand command_def = {
    (GOptionEntry[]) {
      { "string", 0, 0, G_OPTION_ARG_STRING },
      { 0 }
    },
    test_command,
    NULL,
    test_complete5_complete,
  };

  g_autoptr(FoundryCliCommandTree) tree = foundry_cli_command_tree_new ();
  g_autoptr(FoundryCliOptions) options = NULL;
  g_auto(GStrv) comp = NULL;

  foundry_cli_command_tree_register (tree,
                                     FOUNDRY_STRV_INIT ("foundry", "test"),
                                     &command_def);

  comp = foundry_cli_command_tree_complete (tree, command_line, "foundry test --string ", 22, "");

  g_assert_nonnull (comp);
  g_assert_cmpint (g_strv_length (comp), ==, 3);
  g_assert_cmpstr (comp[0], ==, "first");
  g_assert_cmpstr (comp[1], ==, "second");
  g_assert_cmpstr (comp[2], ==, "third");
  g_assert_null (comp[3]);
}

int
main (int   argc,
      char *argv[])
{
  g_test_init (&argc, &argv, NULL);
  command_line = FOUNDRY_COMMAND_LINE (foundry_command_line_local_new ());
  g_test_add_func ("/Foundry/CliCommand/tree1", test_tree1);
  g_test_add_func ("/Foundry/CliCommand/tree2", test_tree2);
  g_test_add_func ("/Foundry/CliCommand/tree3", test_tree3);
  g_test_add_func ("/Foundry/CliCommand/complete1", test_complete1);
  g_test_add_func ("/Foundry/CliCommand/complete2", test_complete2);
  g_test_add_func ("/Foundry/CliCommand/complete3", test_complete3);
  g_test_add_func ("/Foundry/CliCommand/complete4", test_complete4);
  g_test_add_func ("/Foundry/CliCommand/complete5", test_complete5);
  return g_test_run ();
}
