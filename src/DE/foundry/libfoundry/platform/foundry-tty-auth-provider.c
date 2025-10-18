/* foundry-tty-auth-provider.c
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

#include <glib/gstdio.h>

#include "foundry-command-line-input-private.h"
#include "foundry-tty-auth-provider.h"

struct _FoundryTtyAuthProvider
{
  FoundryAuthProvider parent_instance;
  int pty_fd;
};

G_DEFINE_FINAL_TYPE (FoundryTtyAuthProvider, foundry_tty_auth_provider, FOUNDRY_TYPE_AUTH_PROVIDER)

static DexFuture *
foundry_tty_auth_provider_prompt (FoundryAuthProvider *auth_provider,
                                  FoundryInput        *prompt)
{
  FoundryTtyAuthProvider *self = (FoundryTtyAuthProvider *)auth_provider;

  g_assert (FOUNDRY_IS_TTY_AUTH_PROVIDER (self));
  g_assert (FOUNDRY_IS_INPUT (prompt));

  return foundry_command_line_input (self->pty_fd, prompt);
}

static void
foundry_tty_auth_provider_finalize (GObject *object)
{
  FoundryTtyAuthProvider *self = (FoundryTtyAuthProvider *)object;

  g_clear_fd (&self->pty_fd, NULL);

  G_OBJECT_CLASS (foundry_tty_auth_provider_parent_class)->finalize (object);
}

static void
foundry_tty_auth_provider_class_init (FoundryTtyAuthProviderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryAuthProviderClass *auth_provider_class = FOUNDRY_AUTH_PROVIDER_CLASS (klass);

  object_class->finalize = foundry_tty_auth_provider_finalize;

  auth_provider_class->prompt = foundry_tty_auth_provider_prompt;
}

static void
foundry_tty_auth_provider_init (FoundryTtyAuthProvider *self)
{
  self->pty_fd = -1;
}

FoundryAuthProvider *
foundry_tty_auth_provider_new (int pty_fd)
{
  g_return_val_if_fail (pty_fd > -1, NULL);
  g_return_val_if_fail (isatty (pty_fd), NULL);

  pty_fd = dup (pty_fd);

  if (pty_fd > -1)
    {
      FoundryTtyAuthProvider *self = g_object_new (FOUNDRY_TYPE_TTY_AUTH_PROVIDER, NULL);
      self->pty_fd = pty_fd;
      return FOUNDRY_AUTH_PROVIDER (self);
    }

  return NULL;
}
