/* foundry-git-callbacks.c
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

#include <glib/gi18n-lib.h>

#include <libssh2.h>

#include "foundry-git-callbacks-private.h"
#include "foundry-input-group.h"
#include "foundry-input-password.h"
#include "foundry-input-text.h"
#include "foundry-operation.h"

typedef struct _CallbackState
{
  FoundryAuthProvider *auth_provider;
  FoundryOperation    *operation;
  guint                tried;
  int                  pty_fd;
} CallbackState;

static void
ssh_interactive_prompt (const char                            *name,
                        int                                    name_len,
                        const char                            *instruction,
                        int                                    instruction_len,
                        int                                    num_prompts,
                        const LIBSSH2_USERAUTH_KBDINT_PROMPT  *prompts,
                        LIBSSH2_USERAUTH_KBDINT_RESPONSE      *responses,
                        void                                 **abstract)
{
  CallbackState *state = *abstract;
  g_autoptr(FoundryInput) input = NULL;
  g_autoptr(GPtrArray) prompts_ar = NULL;
  g_autofree char *title = NULL;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_AUTH_PROVIDER (state->auth_provider));

  prompts_ar = g_ptr_array_new_with_free_func (g_object_unref);
  title = g_strndup (instruction, instruction_len);

  for (int j = 0; j < num_prompts; j++)
    {
      const char *prompt_text = (const char *)prompts[j].text;
      gboolean hidden = !prompts[j].echo;

      if (hidden)
        g_ptr_array_add (prompts_ar, foundry_input_password_new (prompt_text, NULL, NULL, NULL));
      else
        g_ptr_array_add (prompts_ar, foundry_input_text_new (prompt_text, NULL, NULL, NULL));
    }

  input = foundry_input_group_new (title, NULL, NULL,
                                   (gpointer)prompts_ar->pdata, prompts_ar->len);

  if (!dex_thread_wait_for (foundry_auth_provider_prompt (state->auth_provider, input), NULL))
    return;

  for (int j = 0; j < num_prompts; j++)
    {
      FoundryInput *item = g_ptr_array_index (prompts_ar, j);
      g_autofree char *value = NULL;

      if (FOUNDRY_IS_INPUT_PASSWORD (item))
        value = foundry_input_password_dup_value (FOUNDRY_INPUT_PASSWORD (item));
      else
        value = foundry_input_text_dup_value (FOUNDRY_INPUT_TEXT (item));

      responses[j].text = strdup (value);
      responses[j].length = value ? strlen (value) : 0;
    }
}

static int
credentials_cb (git_cred     **out,
                const char    *url,
                const char    *username_from_url,
                unsigned int   allowed_types,
                void          *payload)
{
  CallbackState *state = payload;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_AUTH_PROVIDER (state->auth_provider));

  allowed_types &= ~state->tried;

  if (allowed_types & GIT_CREDENTIAL_USERNAME)
    {
      state->tried |= GIT_CREDENTIAL_USERNAME;
      return git_cred_username_new (out,
                                    username_from_url ?
                                      username_from_url :
                                      g_get_user_name ());
    }

  if (allowed_types & GIT_CREDENTIAL_SSH_KEY)
    {
      state->tried |= GIT_CREDENTIAL_SSH_KEY;
      return git_cred_ssh_key_from_agent (out,
                                          username_from_url ?
                                            username_from_url :
                                            g_get_user_name ());
    }

  if (allowed_types & GIT_CREDENTIAL_DEFAULT)
    {
      state->tried |= GIT_CREDENTIAL_DEFAULT;
      return git_cred_default_new (out);
    }

  if (allowed_types & GIT_CREDENTIAL_SSH_INTERACTIVE)
    {
      g_autofree char *username = g_strdup (username_from_url);

      state->tried |= GIT_CREDENTIAL_SSH_INTERACTIVE;

      if (username == NULL)
        {
          g_autoptr(FoundryInput) input = foundry_input_text_new (_("Username"), NULL, NULL, g_get_user_name ());
          g_autofree char *value = NULL;

          if (!dex_thread_wait_for (foundry_auth_provider_prompt (state->auth_provider, input), NULL))
            return GIT_PASSTHROUGH;

          if ((value = foundry_input_text_dup_value (FOUNDRY_INPUT_TEXT (input))))
            g_set_str (&username, value);
        }

      return git_cred_ssh_interactive_new (out, username, ssh_interactive_prompt, state);
    }

  if (allowed_types & GIT_CREDENTIAL_USERPASS_PLAINTEXT)
    {
      g_autoptr(GPtrArray) inputs = g_ptr_array_new_with_free_func (g_object_unref);
      g_autoptr(FoundryInput) input = NULL;
      g_autofree char *user_value = NULL;
      g_autofree char *pass_value = NULL;

      state->tried |= GIT_CREDENTIAL_USERPASS_PLAINTEXT;

      g_ptr_array_add (inputs, foundry_input_text_new (_("Username"), NULL, NULL,
                                                       username_from_url ? username_from_url : g_get_user_name ()));
      g_ptr_array_add (inputs, foundry_input_password_new (_("Password"), NULL, NULL, NULL));

      input = foundry_input_group_new (_("Credentials"), NULL, NULL,
                                       (FoundryInput **)inputs->pdata, inputs->len);

      if (!dex_thread_wait_for (foundry_auth_provider_prompt (state->auth_provider, input), NULL))
        return GIT_PASSTHROUGH;

      user_value = foundry_input_text_dup_value (inputs->pdata[0]);
      pass_value = foundry_input_password_dup_value (inputs->pdata[1]);

      return git_cred_userpass_plaintext_new (out, user_value, pass_value);
    }

  return GIT_PASSTHROUGH;
}

static int
transfer_progress_cb (const git_indexer_progress *stats,
                      void                       *payload)
{
  CallbackState *state = payload;

  g_assert (stats != NULL);
  g_assert (state != NULL);

  if (state->operation != NULL)
    {
      double progress = (double)stats->received_objects / (double)stats->total_objects;
      g_autofree char *size = g_strdup_printf ("%"G_GSIZE_FORMAT, stats->received_bytes);
      g_autofree char *message = g_strdup_printf (_("Received %u/%u objects (%s bytes)"),
                                                  stats->received_objects,
                                                  stats->total_objects,
                                                  size);

      foundry_operation_set_subtitle (state->operation, message);
      foundry_operation_set_progress (state->operation, progress);
    }

  return 0;
}

static int
sideband_progress_cb (const char *str,
                      int         len,
                      void       *payload)
{
  CallbackState *state = payload;

  g_assert (state != NULL);

  if (state->pty_fd > -1)
    (void)write (state->pty_fd, str, len);

  return 0;
}

void
_foundry_git_callbacks_init (git_remote_callbacks *callbacks,
                             FoundryOperation     *operation,
                             FoundryAuthProvider  *auth_provider,
                             int                   pty_fd)
{
  CallbackState *state;

  g_return_if_fail (callbacks != NULL);
  g_return_if_fail (FOUNDRY_IS_OPERATION (operation));
  g_return_if_fail (FOUNDRY_IS_AUTH_PROVIDER (auth_provider));

  state = g_new0 (CallbackState, 1);
  state->auth_provider = auth_provider;
  state->operation = operation;
  state->pty_fd = pty_fd;
  state->tried = 0;

  git_remote_init_callbacks (callbacks, GIT_REMOTE_CALLBACKS_VERSION);

  callbacks->credentials = credentials_cb;
  callbacks->transfer_progress = transfer_progress_cb;
  callbacks->sideband_progress = sideband_progress_cb;
  callbacks->payload = state;
}

void
_foundry_git_callbacks_clear (git_remote_callbacks *callbacks)
{
  g_return_if_fail (callbacks != NULL);

  g_free (callbacks->payload);
}
