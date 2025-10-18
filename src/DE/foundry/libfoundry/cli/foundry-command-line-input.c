/* foundry-command-line-input.c
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

#include <errno.h>
#include <termios.h>
#include <unistd.h>

#include <glib/gstdio.h>

#include <gio/gio.h>

#include "foundry-command-line-input-private.h"
#include "foundry-input-choice.h"
#include "foundry-input-combo.h"
#include "foundry-input-file.h"
#include "foundry-input-group.h"
#include "foundry-input-password.h"
#include "foundry-input-switch.h"
#include "foundry-input-text.h"
#include "foundry-path.h"
#include "foundry-string-object-private.h"

typedef struct _Input
{
  FoundryInput *input;
  int           pty_fd;
} Input;

static void
input_free (Input *state)
{
  g_clear_fd (&state->pty_fd, NULL);
  g_clear_object (&state->input);
  g_free (state);
}

G_GNUC_PRINTF (2, 3)
static void
fd_printf (int         fd,
           const char *format,
           ...)
{
  g_autofree char *formatted = NULL;
  va_list args;
  gssize len;

  va_start (args, format);
  len = g_vasprintf (&formatted, format, args);
  va_end (args);

  if (len < 0)
    return;

  (void)write (fd, formatted, len);
}

static int
fd_getchar (int fd)
{
  char c;

  switch (read (fd, &c, 1))
    {
    case 0: return EOF;
    case 1: return c;
    default: return -1;
    }
}

static gboolean
read_password (int         pty_fd,
               const char *prompt,
               char       *buf,
               size_t      buflen)
{
  struct termios oldt, newt;
  int i = 0;
  int c;

  g_assert (pty_fd > -1);
  g_assert (buf != NULL);
  g_assert (buflen > 0);

  fd_printf (pty_fd, "\033[1m%s\033[0m: ", prompt ? prompt : "");

  if (tcgetattr (pty_fd, &oldt) != 0)
    return FALSE;

  newt = oldt;
  newt.c_lflag &= ~ECHO;

  if (tcsetattr (pty_fd, TCSAFLUSH, &newt) != 0)
    return FALSE;

  while ((c = fd_getchar (pty_fd)) != '\n' && c != EOF && i < buflen - 1)
    buf[i++] = c;
  buf[i] = '\0';

  tcsetattr (pty_fd, TCSAFLUSH, &oldt);

  fd_printf (pty_fd, "\n");

  return TRUE;
}

static gboolean
read_entry (int         pty_fd,
            const char *prompt,
            char       *buf,
            size_t      buflen)
{
  int i = 0;
  int c;

  g_assert (pty_fd > -1);
  g_assert (buf != NULL);
  g_assert (buflen > 0);

  fd_printf (pty_fd, "\033[1m%s\033[0m: ", prompt ? prompt : "");

  while ((c = fd_getchar (pty_fd)) != '\n' && c != EOF && i < buflen - 1)
    buf[i++] = c;
  buf[i] = '\0';

  return TRUE;
}

static void
print_title (int           pty_fd,
             FoundryInput *input)
{
  g_autofree char *title = foundry_input_dup_title (input);

  if (title != NULL)
    fd_printf (pty_fd, "\033[1m%s\033[0m\n", title);
}

static void
print_subtitle (int           pty_fd,
                FoundryInput *input)
{
  g_autofree char *subtitle = foundry_input_dup_subtitle (input);

  if (subtitle != NULL)
    fd_printf (pty_fd, "\033[1m%s\033[0m\n", subtitle);
}

static gboolean
string_type_matches (FoundryInputChoice *a,
                     FoundryInputChoice *b)
{
  g_autoptr(GObject) item_a = NULL;
  g_autoptr(GObject) item_b = NULL;

  if (a == NULL || b == NULL)
    return FALSE;

  item_a = foundry_input_choice_dup_item (a);
  item_b = foundry_input_choice_dup_item (b);

  if (item_a == item_b)
    return TRUE;

  if (!FOUNDRY_IS_STRING_OBJECT (item_a) ||
      !FOUNDRY_IS_STRING_OBJECT (item_b))
    return FALSE;

  return g_strcmp0 (foundry_string_object_get_string (FOUNDRY_STRING_OBJECT (item_a)),
                    foundry_string_object_get_string (FOUNDRY_STRING_OBJECT (item_b))) == 0;

}

static gboolean
foundry_command_line_input_recurse (int           pty_fd,
                                    FoundryInput *input)
{
  g_assert (pty_fd > -1);
  g_assert (FOUNDRY_IS_INPUT (input));

  if (FOUNDRY_IS_INPUT_GROUP (input))
    {
      GListModel *model;
      guint n_items;

      print_title (pty_fd, input);
      print_subtitle (pty_fd, input);

      model = foundry_input_group_list_children (FOUNDRY_INPUT_GROUP (input));
      n_items = g_list_model_get_n_items (model);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryInput) child = g_list_model_get_item (model, i);

          if (!foundry_command_line_input_recurse (pty_fd, child))
            return FALSE;
        }

      return TRUE;
    }
  else if (FOUNDRY_IS_INPUT_TEXT (input))
    {
      g_autofree char *title = foundry_input_dup_title (input);
      g_autofree char *original = foundry_input_text_dup_value (FOUNDRY_INPUT_TEXT (input));
      g_autofree char *full_title = NULL;
      g_autofree char *subtitle = foundry_input_dup_subtitle (input);
      char value[512];

      if (original)
        full_title = g_strdup_printf ("%s[%s]", title, original);
      else
        full_title = g_strdup (title);

      if (subtitle)
        fd_printf (pty_fd, "\n%s\n", subtitle);

    again:
      foundry_input_text_set_value (FOUNDRY_INPUT_TEXT (input), original);

      if (read_entry (pty_fd, full_title, value, sizeof value))
        {
          if (value[0] != 0)
            foundry_input_text_set_value (FOUNDRY_INPUT_TEXT (input), value);

          if (!dex_thread_wait_for (foundry_input_validate (input), NULL))
            goto again;

          return TRUE;
        }
    }
  else if (FOUNDRY_IS_INPUT_PASSWORD (input))
    {
      g_autofree char *title = foundry_input_dup_title (input);
      char value[512];

      if (read_password (pty_fd, title, value, sizeof value))
        {
          foundry_input_password_set_value (FOUNDRY_INPUT_PASSWORD (input), value);
          return TRUE;
        }
    }
  else if (FOUNDRY_IS_INPUT_SWITCH (input))
    {
      g_autofree char *title = foundry_input_dup_title (input);
      g_autofree char *full_title = NULL;
      gboolean before = foundry_input_switch_get_value (FOUNDRY_INPUT_SWITCH (input));
      char value[512];

      full_title = g_strdup_printf ("%s[%s]", title, before ? "yes" : "no");

    switch_again:
      if (read_entry (pty_fd, full_title, value, sizeof value))
        {
          if (g_str_equal (value, "yes") ||
              g_str_equal (value, "Yes") ||
              g_str_equal (value, "YES") ||
              g_str_equal (value, "y") ||
              g_str_equal (value, "Y"))
            {
              foundry_input_switch_set_value (FOUNDRY_INPUT_SWITCH (input), TRUE);
              return TRUE;
            }
          else if (g_str_equal (value, "no") ||
                   g_str_equal (value, "No") ||
                   g_str_equal (value, "NO") ||
                   g_str_equal (value, "n") ||
                   g_str_equal (value, "N"))
            {
              foundry_input_switch_set_value (FOUNDRY_INPUT_SWITCH (input), FALSE);
              return TRUE;
            }
          else if (value[0] == 0)
            {
              return TRUE;
            }

          fd_printf (pty_fd, "Please specify [yes|no]\n");
          goto switch_again;
        }
    }
  else if (FOUNDRY_IS_INPUT_COMBO (input))
    {
      g_autoptr(GListModel) choices = foundry_input_combo_list_choices (FOUNDRY_INPUT_COMBO (input));
      g_autoptr(FoundryInputChoice) default_choice = foundry_input_combo_dup_choice (FOUNDRY_INPUT_COMBO (input));
      guint n_items = choices ? g_list_model_get_n_items (choices) : 0;
      g_autofree char *title = foundry_input_dup_title (input);
      g_autofree char *full_title = NULL;
      int match = -1;
      char value[16];

      print_subtitle (pty_fd, input);

      for (guint i = 0; i < n_items; i++)
        {
          g_autoptr(FoundryInputChoice) choice = g_list_model_get_item (choices, i);
          g_autofree char *c_title = foundry_input_dup_title (FOUNDRY_INPUT (choice));
          g_autofree char *c_subtitle = foundry_input_dup_subtitle (FOUNDRY_INPUT (choice));

          if (c_subtitle)
            fd_printf (pty_fd, "%2d: %s (%s)\n", i + 1, c_title, c_subtitle);
          else
            fd_printf (pty_fd, "%2d: %s\n", i + 1, c_title);

          if (default_choice == choice ||
              string_type_matches (default_choice, choice))
            match = i;
        }

      if (match > -1)
        full_title = g_strdup_printf ("%s[%u]", title, match + 1);
      else
        full_title = g_strdup (title);

    combo_again:
      if (read_entry (pty_fd, full_title, value, sizeof value))
        {
          g_autoptr(FoundryInputChoice) choice = NULL;
          char *endptr;
          gint64 n;

          if (value[0] == 0)
            return default_choice != NULL;

          if (!(n = g_ascii_strtoull (value, &endptr, 10)) || *endptr != 0 ||
              n > g_list_model_get_n_items (choices))
            goto combo_again;

          choice = g_list_model_get_item (choices, n - 1);
          foundry_input_combo_set_choice (FOUNDRY_INPUT_COMBO (input), choice);

          return TRUE;
        }
    }
  else if (FOUNDRY_IS_INPUT_FILE (input))
    {
      g_autoptr(GFile) val = foundry_input_file_dup_value (FOUNDRY_INPUT_FILE (input));
      g_autofree char *title = foundry_input_dup_title (input);
      g_autofree char *full_title = NULL;
      char value[512];

      print_subtitle (pty_fd, input);

      if (val != NULL)
        {
          g_autofree char *path = g_file_get_path (val);
          full_title = g_strdup_printf ("%s[%s]", title, path);
        }

      if (read_entry (pty_fd, full_title, value, sizeof value))
        {
          g_autofree char *expand = NULL;
          g_autoptr(GFile) file = NULL;

          if (value[0] == 0)
            return TRUE;

          if (value[0] == '~')
            expand = foundry_path_expand (value);
          else if (g_path_is_absolute (value))
            expand = g_strdup (value);
          else
            expand = g_build_filename (g_get_current_dir (), value, NULL);

          if ((file = g_file_new_for_path (expand)))
            {
              foundry_input_file_set_value (FOUNDRY_INPUT_FILE (input), file);
              return TRUE;
            }
        }
    }

  return FALSE;
}

static DexFuture *
foundry_command_line_input_thread (gpointer data)
{
  Input *state = data;

  g_assert (state != NULL);
  g_assert (state->pty_fd != -1);
  g_assert (FOUNDRY_IS_INPUT (state->input));

  if (!foundry_command_line_input_recurse (state->pty_fd, state->input))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "Failed");

  return dex_future_new_true ();
}

DexFuture *
foundry_command_line_input (int           pty_fd,
                            FoundryInput *input)
{
  Input *state;

  dex_return_error_if_fail (pty_fd > -1);
  dex_return_error_if_fail (FOUNDRY_IS_INPUT (input));

  if (-1 == (pty_fd = dup (pty_fd)))
    return dex_future_new_for_errno (errno);

  state = g_new0 (Input, 1);
  state->pty_fd = pty_fd;
  state->input = g_object_ref (input);

  return dex_thread_spawn ("[foundry-tty-input]",
                           foundry_command_line_input_thread,
                           state,
                           (GDestroyNotify) input_free);
}
