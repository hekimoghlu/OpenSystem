/* foundry-extension-util.c
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

#include <stdlib.h>

#include <gobject/gvaluecollector.h>

#include "foundry-extension-util-private.h"
#include "foundry-util-private.h"

gboolean
foundry_extension_util_can_use_plugin (PeasEngine     *engine,
                                       PeasPluginInfo *plugin_info,
                                       GType           interface_type,
                                       const gchar    *key,
                                       const gchar    *value,
                                       gint           *priority)
{
  g_autofree gchar *path = NULL;
  g_autoptr(GSettings) settings = NULL;

  g_return_val_if_fail (plugin_info != NULL, FALSE);
  g_return_val_if_fail (g_type_is_a (interface_type, G_TYPE_INTERFACE) ||
                        g_type_is_a (interface_type, G_TYPE_OBJECT), FALSE);
  g_return_val_if_fail (priority != NULL, FALSE);

  *priority = 0;

  /*
   * If we are restricting by plugin info keyword, ensure we have enough
   * information to do so.
   */
  if ((key != NULL) && (value == NULL))
    {
      const gchar *found;

      /* If the plugin has the key and its empty, or doesn't have the key,
       * then we can assume it wants the equivalent of "*".
       */
      found = peas_plugin_info_get_external_data (plugin_info, key);
      if (foundry_str_empty0 (found))
        return TRUE;

      return FALSE;
    }

  /*
   * If the plugin isn't loaded, then we shouldn't use it.
   */
  if (!peas_plugin_info_is_loaded (plugin_info))
    return FALSE;

  /*
   * If this plugin doesn't provide this type, we can't use it either.
   */
  if (!peas_engine_provides_extension (engine, plugin_info, interface_type))
    return FALSE;

  /*
   * Check that the plugin provides the match value we are looking for.
   * If key is NULL, then we aren't restricting by matching.
   */
  if (key != NULL)
    {
      g_autofree gchar *priority_name = NULL;
      g_autofree gchar *delimit = NULL;
      g_auto(GStrv) values_array = NULL;
      const gchar *values;
      const gchar *priority_value;

      values = peas_plugin_info_get_external_data (plugin_info, key);
      /* Canonicalize input (for both , and ;) */
      delimit = g_strdelimit (g_strdup (values ? values : ""), ";,", ';');
      values_array = g_strsplit (delimit, ";", 0);

      /* An empty value implies "*" to match anything */
      if (!values || g_strv_contains ((const gchar * const *)values_array, "*"))
        return TRUE;

      /* Otherwise actually check that the key/value matches */
      if (!g_strv_contains ((const gchar * const *)values_array, value))
        return FALSE;

      priority_name = g_strdup_printf ("%s-Priority", key);
      priority_value = peas_plugin_info_get_external_data (plugin_info, priority_name);
      if (priority_value != NULL)
        *priority = atoi (priority_value);
    }

  return TRUE;
}
