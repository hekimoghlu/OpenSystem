/* plugin-flatpak-aux.c
 *
 * Copyright 2021-2025 Christian Hergert <chergert@redhat.com>
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

#include "foundry-util-private.h"

#include "plugin-flatpak-aux.h"

#define SYSTEM_FONTS_DIR       "/usr/share/fonts"
#define SYSTEM_LOCAL_FONTS_DIR "/usr/local/share/fonts"

/* dirs are reversed from flatpak because we will always have
 * /var/cache/fontconfig inside of flatpak. We really need another
 * way of checking this, but this is good enough for now.
 */
#define SYSTEM_FONT_CACHE_DIRS "/var/cache/fontconfig:/usr/lib/fontconfig/cache"

/* The goal of this file is to help us setup things that might be
 * needed for applications to look/work right even though they are
 * not installed. For example, we need to setup font remaps for
 * applications since "flatpak build" will not do that for us.
 */

static GFile *local;
static GFile *mapped;
static GPtrArray *maps;
static gboolean initialized;

static gboolean
query_exists_on_host (const char *path)
{
  g_autofree char *alternate = NULL;

  if (!_foundry_in_container ())
    return g_file_test (path, G_FILE_TEST_EXISTS);

  alternate = g_build_filename ("/var/run/host", path, NULL);

  return g_file_test (alternate, G_FILE_TEST_EXISTS);
}

void
plugin_flatpak_aux_init (void)
{
  g_autoptr(GString) xml_snippet = g_string_new ("");
  g_auto(GStrv) system_cache_dirs = NULL;
  g_autofree char *user1 = NULL;
  g_autofree char *user2 = NULL;
  g_autofree char *user_cache = NULL;
  g_autofree char *cache_dir = NULL;
  g_autofree char *data_dir = NULL;
  guint i;

  if (initialized)
    return;

  initialized = TRUE;

  /* It would be nice if we had a way to get XDG dirs from the host
   * system when we need to break out of flatpak to run flatpak bits
   * through the system.
   */

  if (_foundry_in_container ())
    {
      cache_dir = g_build_filename (g_get_home_dir (), ".cache", NULL);
      data_dir = g_build_filename (g_get_home_dir (), ".local", "share", NULL);
    }
  else
    {
      cache_dir = g_strdup (g_get_user_cache_dir ());
      data_dir = g_strdup (g_get_user_data_dir ());
    }

  local = g_file_new_for_path ("/run/host/font-dirs.xml");
  mapped = g_file_new_build_filename (cache_dir, "font-dirs.xml", NULL);
  maps = g_ptr_array_new ();

  g_string_append (xml_snippet,
                   "<?xml version=\"1.0\"?>\n"
                   "<!DOCTYPE fontconfig SYSTEM \"urn:fontconfig:fonts.dtd\">\n"
                   "<fontconfig>\n");

  if (query_exists_on_host (SYSTEM_FONTS_DIR))
    {
      /* TODO: How can we *force* this read-only? */
      g_ptr_array_add (maps, g_strdup ("--bind-mount=/run/host/fonts=" SYSTEM_FONTS_DIR));
      g_string_append_printf (xml_snippet,
                              "\t<remap-dir as-path=\"%s\">/run/host/fonts</remap-dir>\n",
                              SYSTEM_FONTS_DIR);
    }

  if (query_exists_on_host (SYSTEM_LOCAL_FONTS_DIR))
    {
      /* TODO: How can we *force* this read-only? */
      g_ptr_array_add (maps, g_strdup ("--bind-mount=/run/host/local-fonts=/usr/local/share/fonts"));
      g_string_append_printf (xml_snippet,
                              "\t<remap-dir as-path=\"%s\">/run/host/local-fonts</remap-dir>\n",
                              "/usr/local/share/fonts");
    }

  system_cache_dirs = g_strsplit (SYSTEM_FONT_CACHE_DIRS, ":", 0);
  for (i = 0; system_cache_dirs[i] != NULL; i++)
    {
      if (query_exists_on_host (system_cache_dirs[i]))
        {
          /* TODO: How can we *force* this read-only? */
          g_ptr_array_add (maps,
                           g_strdup_printf ("--bind-mount=/run/host/fonts-cache=%s",
                                            system_cache_dirs[i]));
          break;
        }
    }

  user1 = g_build_filename (data_dir, "fonts", NULL);
  user2 = g_build_filename (g_get_home_dir (), ".fonts", NULL);
  user_cache = g_build_filename (cache_dir, "fontconfig", NULL);

  if (query_exists_on_host (user1))
    {
      g_ptr_array_add (maps, g_strdup_printf ("--filesystem=%s:ro", user1));
      g_string_append_printf (xml_snippet,
                              "\t<remap-dir as-path=\"%s\">/run/host/user-fonts</remap-dir>\n",
                              user1);

    }
  else if (query_exists_on_host (user2))
    {
      g_ptr_array_add (maps, g_strdup_printf ("--filesystem=%s:ro", user2));
      g_string_append_printf (xml_snippet,
                              "\t<remap-dir as-path=\"%s\">/run/host/user-fonts</remap-dir>\n",
                              user2);
    }

  if (query_exists_on_host (user_cache))
    {
      g_ptr_array_add (maps, g_strdup_printf ("--filesystem=%s:ro", user_cache));
      g_ptr_array_add (maps, g_strdup_printf ("--bind-mount=/run/host/user-fonts-cache=%s", user_cache));
    }

  g_string_append (xml_snippet, "</fontconfig>\n");

  g_file_replace_contents (mapped, xml_snippet->str, xml_snippet->len,
                           NULL, FALSE, G_FILE_CREATE_REPLACE_DESTINATION,
                           NULL, NULL, NULL);

  g_ptr_array_add (maps,
                   g_strdup_printf ("--bind-mount=/run/host/font-dirs.xml=%s",
                                    g_file_peek_path (mapped)));
}

void
plugin_flatpak_aux_append_to_launcher (FoundryProcessLauncher *launcher)
{
  static const char *font_dirs_arg;
  gboolean in_container;

  g_return_if_fail (FOUNDRY_IS_PROCESS_LAUNCHER (launcher));
  g_return_if_fail (initialized);

  if (font_dirs_arg == NULL)
    font_dirs_arg = g_strdup_printf ("--bind-mount=/run/host/font-dirs.xml=%s",
                                     g_file_peek_path (mapped));

  for (guint i = 0; i < maps->len; i++)
    {
      const char *element = g_ptr_array_index (maps, i);
      foundry_process_launcher_append_argv (launcher, element);
    }

  foundry_process_launcher_append_argv (launcher, font_dirs_arg);

  in_container = _foundry_in_container ();

  if ((in_container && g_file_test ("/var/run/host/usr/share/icons", G_FILE_TEST_EXISTS)) ||
      (!in_container && g_file_test ("/usr/share/icons", G_FILE_TEST_EXISTS)))
    foundry_process_launcher_append_argv (launcher, "--bind-mount=/run/host/share/icons=/usr/share/icons");

}
