/* foundry-terminal-launcher.c
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
#include <unistd.h>

#include <glib/gstdio.h>

#include "foundry-build-manager.h"
#include "foundry-build-pipeline.h"
#include "foundry-command.h"
#include "foundry-process-launcher.h"
#include "foundry-terminal-launcher.h"
#include "foundry-util.h"

struct _FoundryTerminalLauncher
{
  GObject          parent_instance;
  FoundryCommand  *command;
  char           **override_environment;
} FoundryTerminalLauncherPrivate;

enum {
  PROP_0,
  PROP_COMMAND,
  PROP_OVERRIDE_ENVIRONMENT,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryTerminalLauncher, foundry_terminal_launcher, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_terminal_launcher_dispose (GObject *object)
{
  FoundryTerminalLauncher *self = (FoundryTerminalLauncher *)object;

  g_clear_object (&self->command);
  g_clear_pointer (&self->override_environment, g_strfreev);

  G_OBJECT_CLASS (foundry_terminal_launcher_parent_class)->dispose (object);
}

static void
foundry_terminal_launcher_get_property (GObject    *object,
                                        guint       prop_id,
                                        GValue     *value,
                                        GParamSpec *pspec)
{
  FoundryTerminalLauncher *self = FOUNDRY_TERMINAL_LAUNCHER (object);

  switch (prop_id)
    {
    case PROP_COMMAND:
      g_value_take_object (value, foundry_terminal_launcher_dup_command (self));
      break;

    case PROP_OVERRIDE_ENVIRONMENT:
      g_value_take_boxed (value, foundry_terminal_launcher_dup_override_environment (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_terminal_launcher_set_property (GObject      *object,
                                        guint         prop_id,
                                        const GValue *value,
                                        GParamSpec   *pspec)
{
  FoundryTerminalLauncher *self = FOUNDRY_TERMINAL_LAUNCHER (object);

  switch (prop_id)
    {
    case PROP_COMMAND:
      self->command = g_value_dup_object (value);
      break;

    case PROP_OVERRIDE_ENVIRONMENT:
      self->override_environment = g_value_dup_boxed (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_terminal_launcher_class_init (FoundryTerminalLauncherClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_terminal_launcher_dispose;
  object_class->get_property = foundry_terminal_launcher_get_property;
  object_class->set_property = foundry_terminal_launcher_set_property;

  properties[PROP_COMMAND] =
    g_param_spec_object ("command", NULL, NULL,
                         FOUNDRY_TYPE_COMMAND,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_OVERRIDE_ENVIRONMENT] =
    g_param_spec_boxed ("override-environment", NULL, NULL,
                        G_TYPE_STRV,
                        (G_PARAM_READWRITE |
                         G_PARAM_CONSTRUCT_ONLY |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_terminal_launcher_init (FoundryTerminalLauncher *self)
{
}

/**
 * foundry_terminal_launcher_new:
 * @command:
 * @override_environment: (nullable):
 *
 * Returns: (transfer full):
 */
FoundryTerminalLauncher *
foundry_terminal_launcher_new (FoundryCommand     *command,
                               const char * const *override_environment)
{
  g_return_val_if_fail (FOUNDRY_IS_COMMAND (command), NULL);

  return g_object_new (FOUNDRY_TYPE_TERMINAL_LAUNCHER,
                       "command", command,
                       "override-environment", override_environment,
                       NULL);
}

/**
 * foundry_terminal_launcher_dup_command:
 * @self: a [class@Foundry.TerminalLauncher]
 *
 * Returns: (transfer full):
 */
FoundryCommand *
foundry_terminal_launcher_dup_command (FoundryTerminalLauncher *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TERMINAL_LAUNCHER (self), NULL);
  g_return_val_if_fail (self->command != NULL, NULL);

  return g_object_ref (self->command);
}

/**
 * foundry_terminal_launcher_dup_override_environment:
 * @self: a [class@Foundry.TerminalLauncher]
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_terminal_launcher_dup_override_environment (FoundryTerminalLauncher *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TERMINAL_LAUNCHER (self), NULL);

  return g_strdupv (self->override_environment);
}

typedef struct _Run
{
  FoundryCommand *command;
  int             pty_fd;
} Run;

static void
run_free (Run *state)
{
  g_clear_object (&state->command);
  g_clear_fd (&state->pty_fd, NULL);
  g_free (state);
}

static DexFuture *
foundry_terminal_launcher_run_fiber (gpointer data)
{
  Run *state = data;
  g_autoptr(FoundryProcessLauncher) launcher = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryBuildPipeline) pipeline = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GError) error = NULL;
  guint phase = 0;

  g_assert (state != NULL);
  g_assert (FOUNDRY_IS_COMMAND (state->command));
  g_assert (state->pty_fd > -1);

  launcher = foundry_process_launcher_new ();

  if (!(context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (state->command))))
    return foundry_future_new_disposed ();

  build_manager = foundry_context_dup_build_manager (context);

  if ((pipeline = dex_await_object (foundry_build_manager_load_pipeline (build_manager), NULL)))
    phase = FOUNDRY_BUILD_PIPELINE_PHASE_BUILD;

  if (!dex_await (foundry_command_prepare (state->command, pipeline, launcher, phase), &error))
    return dex_future_new_for_error (g_steal_pointer (&error));

  foundry_process_launcher_set_pty_fd (launcher, state->pty_fd);

  if (!(subprocess = foundry_process_launcher_spawn (launcher, &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_take_object (g_steal_pointer (&subprocess));
}

/**
 * foundry_terminal_launcher_run:
 * @self: a [class@Foundry.TerminalLauncher]
 * @pty_fd: the consumer side of the PTY (e.g. "master")
 *
 * The `pty_fd` is duplicated and therefore may be closed by the
 * caller after calling this function.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@Gio.Subprocess]
 */
DexFuture *
foundry_terminal_launcher_run (FoundryTerminalLauncher *self,
                               int                      pty_fd)
{
  g_autofd int copy = -1;
  Run *state;

  dex_return_error_if_fail (FOUNDRY_IS_TERMINAL_LAUNCHER (self));
  dex_return_error_if_fail (pty_fd > -1);

  if (-1 == (copy = dup (pty_fd)))
    return dex_future_new_for_errno (errno);

  state = g_new0 (Run, 1);
  state->pty_fd = g_steal_fd (&copy);
  state->command = g_object_ref (self->command);

  /* Always spawn the process from the main thread so that we can
   * use prctl() with PDEATHSIG even if our calling thread here
   * is destroyed.
   */
  return dex_scheduler_spawn (dex_scheduler_get_default (), 0,
                              foundry_terminal_launcher_run_fiber,
                              state,
                              (GDestroyNotify) run_free);

}

/**
 * foundry_terminal_launcher_copy:
 * @self: a [class@Foundry.TerminalLauncher]
 *
 * Returns: (transfer full):
 */
FoundryTerminalLauncher *
foundry_terminal_launcher_copy (FoundryTerminalLauncher *self)
{
  g_return_val_if_fail (FOUNDRY_IS_TERMINAL_LAUNCHER (self), NULL);

  return g_object_new (FOUNDRY_TYPE_TERMINAL_LAUNCHER,
                       "command", self->command,
                       "override-environ", self->override_environment,
                       NULL);
}
