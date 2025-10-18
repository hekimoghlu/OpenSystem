/* foundry-git-cloner.c
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
#include <glib/gstdio.h>

#include <git2.h>

#include "foundry-git-autocleanups.h"
#include "foundry-git-callbacks-private.h"
#include "foundry-git-cloner.h"
#include "foundry-git-error.h"
#include "foundry-git-private.h"
#include "foundry-operation.h"
#include "foundry-tty-auth-provider.h"
#include "foundry-util.h"

struct _FoundryGitCloner
{
  GObject  parent_instance;
  char    *author_name;
  char    *author_email;
  char    *remote_branch_name;
  char    *uri;
  GFile   *directory;
  guint    bare : 1;
};

enum {
  PROP_0,
  PROP_AUTHOR_NAME,
  PROP_AUTHOR_EMAIL,
  PROP_BARE,
  PROP_DIRECTORY,
  PROP_REMOTE_BRANCH_NAME,
  PROP_URI,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryGitCloner, foundry_git_cloner, G_TYPE_OBJECT)
G_DEFINE_QUARK (foundry-git-clone-error, foundry_git_clone_error)

static GParamSpec *properties[N_PROPS];

static void
foundry_git_cloner_finalize (GObject *object)
{
  FoundryGitCloner *self = (FoundryGitCloner *)object;

  g_clear_pointer (&self->author_name, g_free);
  g_clear_pointer (&self->author_email, g_free);
  g_clear_pointer (&self->uri, g_free);
  g_clear_pointer (&self->remote_branch_name, g_free);
  g_clear_pointer (&self->uri, g_free);
  g_clear_object (&self->directory);

  G_OBJECT_CLASS (foundry_git_cloner_parent_class)->finalize (object);
}

static void
foundry_git_cloner_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  FoundryGitCloner *self = FOUNDRY_GIT_CLONER (object);

  switch (prop_id)
    {
    case PROP_AUTHOR_NAME:
      g_value_take_string (value, foundry_git_cloner_dup_author_name (self));
      break;

    case PROP_AUTHOR_EMAIL:
      g_value_take_string (value, foundry_git_cloner_dup_author_email (self));
      break;

    case PROP_DIRECTORY:
      g_value_take_object (value, foundry_git_cloner_dup_directory (self));
      break;

    case PROP_REMOTE_BRANCH_NAME:
      g_value_take_string (value, foundry_git_cloner_dup_remote_branch_name (self));
      break;

    case PROP_URI:
      g_value_take_string (value, foundry_git_cloner_dup_uri (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_git_cloner_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  FoundryGitCloner *self = FOUNDRY_GIT_CLONER (object);

  switch (prop_id)
    {
    case PROP_AUTHOR_NAME:
      foundry_git_cloner_set_author_name (self, g_value_get_string (value));
      break;

    case PROP_AUTHOR_EMAIL:
      foundry_git_cloner_set_author_email (self, g_value_get_string (value));
      break;

    case PROP_DIRECTORY:
      foundry_git_cloner_set_directory (self, g_value_get_object (value));
      break;

    case PROP_REMOTE_BRANCH_NAME:
      foundry_git_cloner_set_remote_branch_name (self, g_value_get_string (value));
      break;

    case PROP_URI:
      foundry_git_cloner_set_uri (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_git_cloner_class_init (FoundryGitClonerClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_git_cloner_finalize;
  object_class->get_property = foundry_git_cloner_get_property;
  object_class->set_property = foundry_git_cloner_set_property;

  properties[PROP_AUTHOR_NAME] =
    g_param_spec_string ("author-name", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_AUTHOR_EMAIL] =
    g_param_spec_string ("author-email", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_BARE] =
    g_param_spec_boolean ("bare", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_DIRECTORY] =
    g_param_spec_object ("directory", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_REMOTE_BRANCH_NAME] =
    g_param_spec_string ("remote-branch-name", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_URI] =
    g_param_spec_string ("uri", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_git_cloner_init (FoundryGitCloner *self)
{
  _foundry_git_init ();
}

char *
foundry_git_cloner_dup_uri (FoundryGitCloner *self)
{
  g_return_val_if_fail (FOUNDRY_IS_GIT_CLONER (self), NULL);

  return g_strdup (self->uri);
}

void
foundry_git_cloner_set_uri (FoundryGitCloner *self,
                            const char       *uri)
{
  g_return_if_fail (FOUNDRY_IS_GIT_CLONER (self));

  if (g_set_str (&self->uri, uri))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_URI]);
}

/**
 * foundry_git_cloner_dup_directory:
 * @self: a [class@Foundry.GitCloner]
 *
 * Returns: (transfer full) (nullable):
 */
GFile *
foundry_git_cloner_dup_directory (FoundryGitCloner *self)
{
  g_return_val_if_fail (FOUNDRY_IS_GIT_CLONER (self), NULL);

  return self->directory ? g_object_ref (self->directory) : NULL;
}

void
foundry_git_cloner_set_directory (FoundryGitCloner *self,
                                  GFile            *directory)
{
  g_return_if_fail (FOUNDRY_IS_GIT_CLONER (self));
  g_return_if_fail (!directory || G_IS_FILE (directory));

  if (g_set_object (&self->directory, directory))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_DIRECTORY]);
}

char *
foundry_git_cloner_dup_remote_branch_name (FoundryGitCloner *self)
{
  g_return_val_if_fail (FOUNDRY_IS_GIT_CLONER (self), NULL);

  return g_strdup (self->remote_branch_name);
}

void
foundry_git_cloner_set_remote_branch_name (FoundryGitCloner *self,
                                           const char       *remote_branch_name)
{
  g_return_if_fail (FOUNDRY_IS_GIT_CLONER (self));

  if (g_set_str (&self->remote_branch_name, remote_branch_name))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_REMOTE_BRANCH_NAME]);
}

char *
foundry_git_cloner_dup_author_name (FoundryGitCloner *self)
{
  g_return_val_if_fail (FOUNDRY_IS_GIT_CLONER (self), NULL);

  return g_strdup (self->author_name);
}

void
foundry_git_cloner_set_author_name (FoundryGitCloner *self,
                                    const char       *author_name)
{
  g_return_if_fail (FOUNDRY_IS_GIT_CLONER (self));

  if (g_set_str (&self->author_name, author_name))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_AUTHOR_NAME]);
}

char *
foundry_git_cloner_dup_author_email (FoundryGitCloner *self)
{
  g_return_val_if_fail (FOUNDRY_IS_GIT_CLONER (self), NULL);

  return g_strdup (self->author_email);
}

void
foundry_git_cloner_set_author_email (FoundryGitCloner *self,
                                     const char       *author_email)
{
  g_return_if_fail (FOUNDRY_IS_GIT_CLONER (self));

  if (g_set_str (&self->author_email, author_email))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_AUTHOR_EMAIL]);
}

static DexFuture *
foundry_git_cloner_validate_fiber (gpointer data)
{
  FoundryGitCloner *self = (FoundryGitCloner *)data;

  g_assert (FOUNDRY_IS_GIT_CLONER (self));

  if (self->remote_branch_name != NULL)
    {
      char full_ref[1024];

      if (git_reference_normalize_name (full_ref, sizeof full_ref,
                                        self->remote_branch_name,
                                        GIT_REFERENCE_FORMAT_ALLOW_ONELEVEL) != 0)
        return dex_future_new_reject (FOUNDRY_GIT_CLONE_ERROR,
                                      FOUNDRY_GIT_CLONE_ERROR_INVALID_REMOTE_BRANCH_NAME,
                                      _("Invalid branch name"));
    }

  if (self->directory == NULL)
    return dex_future_new_reject (FOUNDRY_GIT_CLONE_ERROR,
                                  FOUNDRY_GIT_CLONE_ERROR_INVALID_DIRECTORY,
                                  _("Directory must be set"));

  if (dex_await (dex_file_query_exists (self->directory), NULL))
    return dex_future_new_reject (FOUNDRY_GIT_CLONE_ERROR,
                                  FOUNDRY_GIT_CLONE_ERROR_INVALID_DIRECTORY,
                                  _("Directory already exists"));

  return dex_future_new_true ();
}

/**
 * foundry_git_cloner_validate:
 * @self: a [class@Foundry.GitCloner]
 *
 * Validates the values of the cloner.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value or rejects with error
 */
DexFuture *
foundry_git_cloner_validate (FoundryGitCloner *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_GIT_CLONER (self));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_git_cloner_validate_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

typedef struct _Clone
{
  FoundryAuthProvider *auth_provider;
  FoundryOperation    *operation;
  char                *author_name;
  char                *author_email;
  char                *remote_branch_name;
  char                *uri;
  GFile               *directory;
  int                  pty_fd;
  guint                bare : 1;
} Clone;

static void
clone_free (Clone *state)
{
  g_clear_pointer (&state->author_name, g_free);
  g_clear_pointer (&state->author_email, g_free);
  g_clear_pointer (&state->remote_branch_name, g_free);
  g_clear_pointer (&state->uri, g_free);
  g_clear_object (&state->directory);
  g_clear_object (&state->operation);
  g_clear_object (&state->auth_provider);
  g_clear_fd (&state->pty_fd, NULL);
  g_free (state);
}

static DexFuture *
foundry_git_cloner_clone_thread (gpointer data)
{
  Clone *state = data;
  git_clone_options clone_opts = GIT_CLONE_OPTIONS_INIT;
  g_autoptr(git_repository) repository = NULL;
  g_autofree char *path = NULL;
  int rval;

  g_assert (state != NULL);
  g_assert (G_IS_FILE (state->directory));
  g_assert (g_file_is_native (state->directory));

  path = g_file_get_path (state->directory);

  clone_opts.bare = state->bare;
  clone_opts.fetch_opts.download_tags = GIT_REMOTE_DOWNLOAD_TAGS_NONE;

  if (state->remote_branch_name)
    clone_opts.checkout_branch = state->remote_branch_name;

  _foundry_git_callbacks_init (&clone_opts.fetch_opts.callbacks, state->operation, state->auth_provider, state->pty_fd);
  rval = git_clone (&repository, state->uri, path, &clone_opts);
  _foundry_git_callbacks_clear (&clone_opts.fetch_opts.callbacks);

  if (rval != 0)
    return foundry_git_reject_last_error ();

  return dex_future_new_true ();
}

/**
 * foundry_git_cloner_clone:
 * @self: a [class@Foundry.GitCloner]
 * @pty_fd: the FD for a PTY, or -1
 * @operation:
 *
 * @pty_fd is copied and may be closed after calling this function.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   any value or rejects with error
 */
DexFuture *
foundry_git_cloner_clone (FoundryGitCloner *self,
                          int               pty_fd,
                          FoundryOperation *operation)
{
  Clone *state;

  dex_return_error_if_fail (FOUNDRY_IS_GIT_CLONER (self));
  dex_return_error_if_fail (FOUNDRY_IS_OPERATION (operation));

  if (self->uri == NULL)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "Missing URI");

  if (self->directory == NULL || !g_file_is_native (self->directory))
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_FAILED,
                                  "Missing local directory");

  state = g_new0 (Clone, 1);
  state->operation = g_object_ref (operation);
  state->auth_provider = foundry_operation_dup_auth_provider (operation);
  state->author_name = g_strdup (self->author_name);
  state->author_email = g_strdup (self->author_email);
  state->remote_branch_name = g_strdup (self->remote_branch_name);
  state->uri = g_strdup (self->uri);
  state->directory = g_file_dup (self->directory);
  state->pty_fd = dup (pty_fd);
  state->bare = self->bare;

  if (state->auth_provider == NULL)
    state->auth_provider = foundry_tty_auth_provider_new (pty_fd);

  return dex_thread_spawn ("[git-clone]",
                           foundry_git_cloner_clone_thread,
                           state,
                           (GDestroyNotify) clone_free);
}

/**
 * foundry_git_cloner_get_bare:
 * @self: a [class@Foundry.GitCloner]
 *
 * If [property@Foundry.GitCloner:directory] should be used as the destination
 * directory instead of a `.git` subdirectory.
 */
gboolean
foundry_git_cloner_get_bare (FoundryGitCloner *self)
{
  g_return_val_if_fail (FOUNDRY_IS_GIT_CLONER (self), FALSE);

  return self->bare;
}

void
foundry_git_cloner_set_bare (FoundryGitCloner *self,
                             gboolean          bare)
{
  g_return_if_fail (FOUNDRY_IS_GIT_CLONER (self));

  bare = !!bare;

  if (bare != self->bare)
    {
      self->bare = bare;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_BARE]);
    }
}

FoundryGitCloner *
foundry_git_cloner_new (void)
{
  return g_object_new (FOUNDRY_TYPE_GIT_CLONER, NULL);
}
