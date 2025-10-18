/* foundry-cli-command.c
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

#include <glib/gi18n-lib.h>

#include "foundry-cli-command-private.h"
#include "foundry-command-line.h"
#include "foundry-command-line-remote-private.h"
#include "foundry-context.h"
#include "foundry-init-private.h"
#include "foundry-util.h"

G_DEFINE_BOXED_TYPE (FoundryCliOptions,
                     foundry_cli_options,
                     foundry_cli_options_ref,
                     foundry_cli_options_unref)

struct _FoundryCliOptions
{
  GHashTable *hash;
};

static void
_g_value_free (gpointer data)
{
  GValue *value = data;
  g_value_unset (value);
  g_free (value);
}

/**
 * foundry_cli_options_new:
 *
 * Returns: (transfer full): a new #FoundryCliOptions
 */
FoundryCliOptions *
foundry_cli_options_new (void)
{
  FoundryCliOptions *self;

  self = g_atomic_rc_box_new0 (FoundryCliOptions);
  self->hash = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, _g_value_free);

  return self;
}

/**
 * foundry_cli_options_ref: (skip)
 * @self: a #FoundryCliOptions
 *
 * Returns: (transfer full): @self
 */
FoundryCliOptions *
foundry_cli_options_ref (FoundryCliOptions *self)
{
  return g_atomic_rc_box_acquire (self);
}

static void
foundry_cli_options_finalize (gpointer data)
{
  FoundryCliOptions *self = data;

  g_clear_pointer (&self->hash, g_hash_table_unref);
}

void
foundry_cli_options_unref (FoundryCliOptions *self)
{
  g_atomic_rc_box_release_full (self, foundry_cli_options_finalize);
}

static DexFuture *
foundry_cli_options_load_context_fiber (const char         *foundry_dir,
                                        FoundryCommandLine *command_line,
                                        gboolean            shared)
{
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_COMMAND_LINE (command_line));

  if (FOUNDRY_IS_COMMAND_LINE_REMOTE (command_line))
    {
      FoundryCommandLineRemote *remote = FOUNDRY_COMMAND_LINE_REMOTE (command_line);
      FoundryContext *context = foundry_command_line_remote_get_context (remote);

      if (context != NULL)
        return dex_future_new_for_object (g_object_ref (context));
    }

  if (shared)
    return foundry_context_new_for_user (NULL);

  if (foundry_dir == NULL)
    {
      const char *directory = foundry_command_line_get_directory (command_line);

      if (!(foundry_dir = dex_await_string (foundry_context_discover (directory, NULL), &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));
    }

  return foundry_context_new (foundry_dir,
                              NULL,
                              FOUNDRY_CONTEXT_FLAGS_NONE,
                              NULL);
}

/**
 * foundry_cli_options_load_context:
 * @self: a #FoundryCliOptions
 *
 * Loads the #FoundryContext based on option values.
 *
 * Returns: (transfer full): a #DexFuture that resolves to a #FoundryContext
 */
DexFuture *
foundry_cli_options_load_context (FoundryCliOptions  *self,
                                  FoundryCommandLine *command_line)
{
  gboolean shared;

  dex_return_error_if_fail (self != NULL);
  dex_return_error_if_fail (FOUNDRY_IS_COMMAND_LINE (command_line));

  _foundry_init_plugins ();

  if (!foundry_cli_options_get_boolean (self, "shared", &shared))
    shared = FALSE;

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (foundry_cli_options_load_context_fiber),
                                  3,
                                  G_TYPE_STRING, foundry_cli_options_get_string (self, "foundry-dir"),
                                  FOUNDRY_TYPE_COMMAND_LINE, command_line,
                                  G_TYPE_BOOLEAN, shared);
}

static const GValue *
foundry_cli_options_get (FoundryCliOptions *self,
                         const char        *key,
                         GType              type)
{
  const GValue *value;

  if ((value = g_hash_table_lookup (self->hash, key)) && !G_VALUE_HOLDS (value, type))
    value = NULL;

  return value;
}

static GValue *
foundry_cli_options_set (FoundryCliOptions *self,
                         const char        *key,
                         GType              type)
{
  GValue *value;

  if ((value = g_hash_table_lookup (self->hash, key)))
    {
      g_value_unset (value);
    }
  else
    {
      value = g_new0 (GValue, 1);
      g_hash_table_replace (self->hash, g_strdup (key), value);
    }

  g_value_init (value, type);

  return value;
}

const char *
foundry_cli_options_get_string (FoundryCliOptions *self,
                                const char        *key)
{
  const GValue *gvalue;

  g_return_val_if_fail (self != NULL, NULL);
  g_return_val_if_fail (key != NULL, NULL);

  if ((gvalue = foundry_cli_options_get (self, key, G_TYPE_STRING)))
    return g_value_get_string (gvalue);

  return NULL;
}

void
foundry_cli_options_set_string (FoundryCliOptions *self,
                                const char        *key,
                                const char        *value)
{
  g_return_if_fail (self != NULL);
  g_return_if_fail (key != NULL);

  if (value == NULL)
    g_hash_table_remove (self->hash, key);
  else
    g_value_set_string (foundry_cli_options_set (self, key, G_TYPE_STRING), value);
}

const char *
foundry_cli_options_get_filename (FoundryCliOptions *self,
                                  const char        *key)
{
  const GValue *gvalue;

  g_return_val_if_fail (self != NULL, NULL);
  g_return_val_if_fail (key != NULL, NULL);

  if ((gvalue = foundry_cli_options_get (self, key, G_TYPE_STRING)))
    return g_value_get_string (gvalue);

  return NULL;
}

void
foundry_cli_options_set_filename (FoundryCliOptions *self,
                                  const char        *key,
                                  const char        *value)
{
  g_return_if_fail (self != NULL);
  g_return_if_fail (key != NULL);

  if (value == NULL)
    g_hash_table_remove (self->hash, key);
  else
    g_value_set_string (foundry_cli_options_set (self, key, G_TYPE_STRING), value);
}

const char * const *
foundry_cli_options_get_string_array (FoundryCliOptions *self,
                                      const char        *key)
{
  const GValue *gvalue;

  g_return_val_if_fail (self != NULL, NULL);
  g_return_val_if_fail (key != NULL, NULL);

  if ((gvalue = foundry_cli_options_get (self, key, G_TYPE_STRV)))
    return g_value_get_boxed (gvalue);

  return NULL;
}

void
foundry_cli_options_set_string_array (FoundryCliOptions  *self,
                                      const char         *key,
                                      const char * const *value)
{
  g_return_if_fail (self != NULL);
  g_return_if_fail (key != NULL);

  if (value == NULL)
    g_hash_table_remove (self->hash, key);
  else
    g_value_set_boxed (foundry_cli_options_set (self, key, G_TYPE_STRV), value);
}

const char * const *
foundry_cli_options_get_filename_array (FoundryCliOptions *self,
                                        const char        *key)
{
  const GValue *gvalue;

  g_return_val_if_fail (self != NULL, NULL);
  g_return_val_if_fail (key != NULL, NULL);

  if ((gvalue = foundry_cli_options_get (self, key, G_TYPE_STRV)))
    return g_value_get_boxed (gvalue);

  return NULL;
}

void
foundry_cli_options_set_filename_array (FoundryCliOptions  *self,
                                        const char         *key,
                                        const char * const *value)
{
  g_return_if_fail (self != NULL);
  g_return_if_fail (key != NULL);

  if (value == NULL)
    g_hash_table_remove (self->hash, key);
  else
    g_value_set_boxed (foundry_cli_options_set (self, key, G_TYPE_STRV), value);
}

gboolean
foundry_cli_options_get_int (FoundryCliOptions *self,
                             const char        *key,
                             int               *value)
{
  const GValue *gvalue;

  g_return_val_if_fail (self != NULL, FALSE);
  g_return_val_if_fail (key != NULL, FALSE);

  if ((gvalue = foundry_cli_options_get (self, key, G_TYPE_INT)))
    {
      if (value != NULL)
        *value = g_value_get_int (gvalue);
      return TRUE;
    }

  return FALSE;
}

void
foundry_cli_options_set_int (FoundryCliOptions *self,
                             const char        *key,
                             int                value)
{
  g_return_if_fail (self != NULL);
  g_return_if_fail (key != NULL);

  g_value_set_int (foundry_cli_options_set (self, key, G_TYPE_INT), value);
}

gboolean
foundry_cli_options_get_int64 (FoundryCliOptions *self,
                               const char        *key,
                               gint64            *value)
{
  const GValue *gvalue;

  g_return_val_if_fail (self != NULL, FALSE);
  g_return_val_if_fail (key != NULL, FALSE);

  if ((gvalue = foundry_cli_options_get (self, key, G_TYPE_INT64)))
    {
      if (value != NULL)
        *value = g_value_get_int64 (gvalue);
      return TRUE;
    }

  return FALSE;
}

void
foundry_cli_options_set_int64 (FoundryCliOptions *self,
                               const char        *key,
                               gint64             value)
{
  g_return_if_fail (self != NULL);
  g_return_if_fail (key != NULL);

  g_value_set_int64 (foundry_cli_options_set (self, key, G_TYPE_INT64), value);
}

gboolean
foundry_cli_options_get_double (FoundryCliOptions *self,
                                const char        *key,
                                double            *value)
{
  const GValue *gvalue;

  g_return_val_if_fail (self != NULL, FALSE);
  g_return_val_if_fail (key != NULL, FALSE);

  if ((gvalue = foundry_cli_options_get (self, key, G_TYPE_DOUBLE)))
    {
      if (value != NULL)
        *value = g_value_get_double (gvalue);
      return TRUE;
    }

  return FALSE;
}

void
foundry_cli_options_set_double (FoundryCliOptions *self,
                                const char        *key,
                                double             value)
{
  g_return_if_fail (self != NULL);
  g_return_if_fail (key != NULL);

  g_value_set_double (foundry_cli_options_set (self, key, G_TYPE_DOUBLE), value);
}

gboolean
foundry_cli_options_get_boolean (FoundryCliOptions *self,
                                 const char        *key,
                                 gboolean          *value)
{
  const GValue *gvalue;

  g_return_val_if_fail (self != NULL, FALSE);
  g_return_val_if_fail (key != NULL, FALSE);

  if ((gvalue = foundry_cli_options_get (self, key, G_TYPE_BOOLEAN)))
    {
      if (value != NULL)
        *value = g_value_get_boolean (gvalue);
      return TRUE;
    }

  return FALSE;
}

void
foundry_cli_options_set_boolean (FoundryCliOptions *self,
                                 const char        *key,
                                 gboolean           value)
{
  g_return_if_fail (self != NULL);
  g_return_if_fail (key != NULL);

  g_value_set_boolean (foundry_cli_options_set (self, key, G_TYPE_BOOLEAN), value);
}

static gsize
count_option_entries (const GOptionEntry *entries)
{
  guint i;

  if (entries == NULL)
    return 0;

  for (i = 0; entries[i].long_name; i++) { }

  return i;
}

static GOptionEntry *
copy_option_entries (const GOptionEntry *entries)
{
  GOptionEntry *copy;

  if (entries == NULL)
    return NULL;

  copy = g_memdup2 (entries, sizeof (GOptionEntry) * (count_option_entries (entries) + 1));

  for (guint i = 0; copy[i].long_name; i++)
    {
      copy[i].long_name = g_strdup (copy[i].long_name);
      copy[i].description = g_strdup (copy[i].description);
      copy[i].arg_description = g_strdup (copy[i].arg_description);
    }

  return copy;
}

static void
free_option_entries (GOptionEntry *entries)
{
  if (entries == NULL)
    return;

  for (guint i = 0; entries[i].long_name; i++)
    {
      g_free ((char *)entries[i].long_name);
      g_free ((char *)entries[i].description);
      g_free ((char *)entries[i].arg_description);
    }

  g_free (entries);
}

FoundryCliCommand *
foundry_cli_command_copy (const FoundryCliCommand *command)
{
  FoundryCliCommand *copy;

  g_return_val_if_fail (command != NULL, NULL);

  copy = g_memdup2 (command, sizeof *command);
  copy->gettext_package = g_strdup (command->gettext_package);
  copy->description = g_strdup (command->description);
  copy->options = copy_option_entries (command->options);

  return copy;
}

void
foundry_cli_command_free (FoundryCliCommand *command)
{
  g_clear_pointer (&command->options, free_option_entries);
  g_clear_pointer ((char **)&command->gettext_package, g_free);
  g_clear_pointer ((char **)&command->description, g_free);
  g_free (command);
}

typedef struct
{
  FoundryCliCommand   *command;
  DexCancellable      *cancellable;
  FoundryCommandLine  *command_line;
  FoundryCliOptions   *options;
  char               **argv;
} Run;

static void
run_free (Run *state)
{
  dex_clear (&state->cancellable);
  g_clear_object (&state->command_line);
  g_clear_pointer (&state->options, foundry_cli_options_unref);
  g_clear_pointer (&state->argv, g_strfreev);
  g_clear_pointer (&state->command, foundry_cli_command_free);
  g_free (state);
}

static DexFuture *
foundry_cli_command_run_fiber (gpointer user_data)
{
  Run *state = user_data;
  int res = EXIT_SUCCESS;

  g_assert (state != NULL);
  g_assert (state->argv != NULL);
  g_assert (state->options != NULL);
  g_assert (state->command != NULL);
  g_assert (!state->cancellable || DEX_IS_CANCELLABLE (state->cancellable));

  if (state->command->run == NULL)
    res = EXIT_FAILURE;

  if (state->command->run == NULL || foundry_cli_options_help (state->options))
    {
      const char *gettext_package = state->command->gettext_package ? state->command->gettext_package : GETTEXT_PACKAGE;
      g_autoptr(GOptionContext) context = NULL;
      g_autofree char *help = NULL;
      const char *skipped;
      const char *argv0;

      context = g_option_context_new (NULL);
      g_option_context_set_help_enabled (context, FALSE);

      if (state->command->options != NULL)
        g_option_context_add_main_entries (context, state->command->options, gettext_package);

      help = g_option_context_get_help (context, TRUE, NULL);

      argv0 = strstr (state->argv[0], "foundry");

      foundry_command_line_printerr (state->command_line, "%s:\n  %s [%s…]", _("Usage"), argv0, _("OPTIONS"));
      if (state->command->description)
        foundry_command_line_printerr (state->command_line,
                                       " %s",
                                       g_dgettext (gettext_package, state->command->description));
      foundry_command_line_printerr (state->command_line, "\n");

      if ((skipped = strstr (help, "]\n")))
        foundry_command_line_printerr (state->command_line, "%s", skipped + strlen ("]\n"));
    }
  else
    {
      res = state->command->run (state->command_line,
                                 (const char * const *)state->argv,
                                 state->options,
                                 state->cancellable);
    }

  return dex_future_new_for_int (res);
}

DexFuture *
foundry_cli_command_run (const FoundryCliCommand *command,
                         FoundryCommandLine      *command_line,
                         const char * const      *argv,
                         FoundryCliOptions       *options,
                         DexCancellable          *cancellable)
{
  Run *state;

  g_return_val_if_fail (command != NULL, NULL);
  g_return_val_if_fail (FOUNDRY_IS_COMMAND_LINE (command_line), NULL);
  g_return_val_if_fail (argv != NULL, NULL);
  g_return_val_if_fail (options != NULL, NULL);
  g_return_val_if_fail (!cancellable || DEX_IS_CANCELLABLE (cancellable), NULL);

  state = g_new0 (Run, 1);
  state->command_line = g_object_ref (command_line);
  state->argv = g_strdupv ((char **)argv);
  state->options = foundry_cli_options_ref (options);
  state->cancellable = cancellable ? dex_ref (cancellable) : NULL;
  state->command = foundry_cli_command_copy (command);

  return dex_scheduler_spawn (NULL, 0,
                              foundry_cli_command_run_fiber,
                              state,
                              (GDestroyNotify) run_free);
}

gboolean
foundry_cli_options_help (FoundryCliOptions *self)
{
  gboolean help = FALSE;

  if (foundry_cli_options_get_boolean (self, "help", &help))
    return help;

  return FALSE;
}
