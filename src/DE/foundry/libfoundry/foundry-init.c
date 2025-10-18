/* foundry-init.c
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
#include <libdex.h>
#include <libpeas.h>
#include <sysprof-capture.h>

#include <json-glib/json-glib.h>

#include "foundry-build-manager.h"
#include "foundry-cli-builtin-private.h"
#include "foundry-cli-command-tree.h"
#include "foundry-command-line-local-private.h"
#include "foundry-command-line-remote-private.h"
#include "foundry-command-line.h"
#include "foundry-config-manager.h"
#include "foundry-config-provider.h"
#include "foundry-config.h"
#include "foundry-context.h"
#include "foundry-debug.h"
#include "foundry-init-private.h"
#include "foundry-plugin-manager.h"
#include "foundry-process-launcher.h"
#include "foundry-shell-private.h"
#include "foundry-triplet-private.h"
#include "foundry-unix-fd-map.h"
#include "foundry-util-private.h"

#include "foundry-resources.h"
#include "foundry-plugins-resources.h"

#include "gconstructor.h"

static GThread *main_thread;
static DexPromise *init_promise;

static void
triplet_to_string (const GValue *src,
                   GValue       *dest)
{
  FoundryTriplet *triplet;

  if ((triplet = g_value_get_boxed (src)))
    g_value_set_string (dest, foundry_triplet_get_full_name (triplet));
}

static DexFuture *
_foundry_init_shell_cb (DexFuture *completed,
                        gpointer   user_data)
{
  DexPromise *promise = user_data;
  dex_promise_resolve_boolean (promise, TRUE);
  return dex_future_new_true ();
}

void
_foundry_init_plugins (void)
{
  static gsize initialized;

  if (g_once_init_enter (&initialized))
    {
      (void)foundry_plugin_manager_get_default ();
      g_once_init_leave (&initialized, TRUE);
    }
}

void
_foundry_init_cli (void)
{
  static gsize initialized;

  if (g_once_init_enter (&initialized))
    {
      FoundryCliCommandTree *tree = foundry_cli_command_tree_get_default ();
      _foundry_cli_builtin_register (tree);
      g_once_init_leave (&initialized, TRUE);
    }
}

static void
_foundry_init (void)
{
  DexFuture *future;
  PeasEngine *engine;

  dex_init ();

  bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
  bindtextdomain (GETTEXT_PACKAGE, LOCALEDIR);

  main_thread = g_thread_self ();

  future = _foundry_shell_init ();
  future = dex_future_finally (future,
                               _foundry_init_shell_cb,
                               dex_ref (init_promise),
                               dex_unref);
  dex_future_disown (future);

  g_type_ensure (FOUNDRY_TYPE_BUILD_MANAGER);
  g_type_ensure (FOUNDRY_TYPE_COMMAND_LINE);
  g_type_ensure (FOUNDRY_TYPE_COMMAND_LINE_LOCAL);
  g_type_ensure (FOUNDRY_TYPE_COMMAND_LINE_REMOTE);
  g_type_ensure (FOUNDRY_TYPE_CONFIG);
  g_type_ensure (FOUNDRY_TYPE_CONFIG_MANAGER);
  g_type_ensure (FOUNDRY_TYPE_CONFIG_PROVIDER);
  g_type_ensure (FOUNDRY_TYPE_CONTEXT);
  g_type_ensure (FOUNDRY_TYPE_CONTEXT_FLAGS);
  g_type_ensure (FOUNDRY_TYPE_PROCESS_LAUNCHER);
  g_type_ensure (FOUNDRY_TYPE_TRIPLET);
  g_type_ensure (FOUNDRY_TYPE_UNIX_FD_MAP);

  g_resources_register (_foundry_get_resource ());
  g_resources_register (_foundry_plugins_get_resource ());

  json_boxed_register_serialize_func (FOUNDRY_TYPE_TRIPLET,
                                      JSON_NODE_VALUE,
                                      _foundry_triplet_to_json);
  g_value_register_transform_func (FOUNDRY_TYPE_TRIPLET,
                                   G_TYPE_STRING,
                                   triplet_to_string);

  _foundry_init_cli ();

  engine = peas_engine_get_default ();
  peas_engine_add_search_path (engine,
                               "resource:///app/devsuite/foundry/plugins",
                               "resource:///app/devsuite/foundry/plugins");
}

/**
 * foundry_init:
 *
 * Initializes libfoundry which is completed once the resulting
 * #DexFuture completes.
 *
 * It is generally fine to use the library after calling this function
 * but some data may not be fully loaded until it resolves. For example,
 * if you need to sniff the user shell or other environment data, await
 * this future before accessing user shell APIs.
 *
 * Returns: (transfer full): a #DexPromise that resolves when the
 *   library and dynamically loaded data has completed.
 */
DexFuture *
foundry_init (void)
{
  FOUNDRY_ENTRY;

  if (g_once_init_enter (&init_promise))
    {
      g_once_init_leave (&init_promise, dex_promise_new ());
      _foundry_init ();
    }

  FOUNDRY_RETURN (dex_ref (DEX_FUTURE (init_promise)));
}

gboolean
foundry_thread_is_main (GThread *thread)
{
  return thread == main_thread;
}

void
foundry_trace_function (const gchar *func,
                        gint64       begin_time_usec,
                        gint64       end_time_usec)
{
  sysprof_collector_mark (begin_time_usec * 1000L,
                          (end_time_usec - begin_time_usec) * 1000L,
                          "tracing",
                          "call",
                          func);
}
