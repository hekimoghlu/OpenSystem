/* plugin-doap-service.c
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

#include "plugin-doap-file.h"
#include "plugin-doap-service.h"

struct _PluginDoapService
{
  FoundryService  parent_instance;
  PluginDoapFile *doap_file;
};

G_DEFINE_FINAL_TYPE (PluginDoapService, plugin_doap_service, FOUNDRY_TYPE_SERVICE)

static DexFuture *
plugin_doap_service_start_fiber (gpointer data)
{
  PluginDoapService *self = data;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GPtrArray) files = NULL;
  g_autoptr(GFile) project_dir = NULL;

  g_assert (PLUGIN_IS_DOAP_SERVICE (self));

  context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (self));
  project_dir = foundry_context_dup_project_directory (context);

  /* Ignore if this isn't a real project */
  if (foundry_context_is_shared (context))
    return dex_future_new_true ();

  g_debug ("Searching `%s` for *.doap project file",
           g_file_peek_path (project_dir));

  /* Find *.doap files so we can parse them */
  if (!(files = dex_await_boxed (foundry_file_find_with_depth (project_dir, "*.doap", 1), NULL)))
    return dex_future_new_true ();

  for (guint i = 0; i < files->len; i++)
    {
      GFile *file = g_ptr_array_index (files, i);
      g_autoptr(GBytes) bytes = NULL;
      g_autoptr(GError) error = NULL;

      if ((bytes = dex_await_boxed (dex_file_load_contents_bytes (file), NULL)))
        {
          g_autoptr(PluginDoapFile) doap_file = plugin_doap_file_new ();

          if (plugin_doap_file_load_from_bytes (doap_file, bytes, &error))
            {
              const char *name = plugin_doap_file_get_name (doap_file);
              g_debug ("Discovered project name \"%s\" from doap.", name);
              g_set_object (&self->doap_file, doap_file);
              foundry_context_set_title (context, name);
              break;
            }
        }
    }

  return dex_future_new_true ();
}

static DexFuture *
plugin_doap_service_start (FoundryService *service)
{
  return dex_scheduler_spawn (NULL, 0,
                              plugin_doap_service_start_fiber,
                              g_object_ref (service),
                              g_object_unref);
}

static DexFuture *
plugin_doap_service_stop (FoundryService *service)
{
  PluginDoapService *self = (PluginDoapService *)service;

  g_assert (PLUGIN_IS_DOAP_SERVICE (self));

  g_clear_object (&self->doap_file);

  return dex_future_new_true ();
}

static void
plugin_doap_service_class_init (PluginDoapServiceClass *klass)
{
  FoundryServiceClass *service_class = FOUNDRY_SERVICE_CLASS (klass);

  service_class->start = plugin_doap_service_start;
  service_class->stop = plugin_doap_service_stop;
}

static void
plugin_doap_service_init (PluginDoapService *self)
{
}
