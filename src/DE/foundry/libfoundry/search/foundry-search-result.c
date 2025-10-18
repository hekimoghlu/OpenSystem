/* foundry-search-result.c
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

#include "foundry-search-result.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundrySearchResult, foundry_search_result, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_ICON,
  PROP_SUBTITLE,
  PROP_TITLE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_search_result_get_property (GObject    *object,
                                    guint       prop_id,
                                    GValue     *value,
                                    GParamSpec *pspec)
{
  FoundrySearchResult *self = FOUNDRY_SEARCH_RESULT (object);

  switch (prop_id)
    {
    case PROP_ICON:
      g_value_take_object (value, foundry_search_result_dup_icon (self));
      break;

    case PROP_SUBTITLE:
      g_value_take_string (value, foundry_search_result_dup_subtitle (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_search_result_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_search_result_class_init (FoundrySearchResultClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_search_result_get_property;

  properties[PROP_ICON] =
    g_param_spec_object ("icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SUBTITLE] =
    g_param_spec_string ("subtitle", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_search_result_init (FoundrySearchResult *self)
{
}

char *
foundry_search_result_dup_title (FoundrySearchResult *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SEARCH_RESULT (self), NULL);

  if (FOUNDRY_SEARCH_RESULT_GET_CLASS (self)->dup_title)
    return FOUNDRY_SEARCH_RESULT_GET_CLASS (self)->dup_title (self);

  return g_strdup (G_OBJECT_TYPE_NAME (self));
}

char *
foundry_search_result_dup_subtitle (FoundrySearchResult *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SEARCH_RESULT (self), NULL);

  if (FOUNDRY_SEARCH_RESULT_GET_CLASS (self)->dup_subtitle)
    return FOUNDRY_SEARCH_RESULT_GET_CLASS (self)->dup_subtitle (self);

  return NULL;
}

/**
 * foundry_search_result_load:
 * @self: a [class@Foundry.SearchResult]
 *
 * Loads the contents of the search result.
 *
 * The consumer of this should know how to handle the specific
 * object type by checking it's `GType`.
 *
 * For example, if the result is a [class@Foundry.Documentation] then
 * you may want to check it's URI property to open the documentation.
 *
 * It is expected that search providers load well known object types
 * which applications can reasonably handle.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to
 *   a [class@GObject.Object].
 */
DexFuture *
foundry_search_result_load (FoundrySearchResult *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_SEARCH_RESULT (self));

  if (FOUNDRY_SEARCH_RESULT_GET_CLASS (self)->load)
    return FOUNDRY_SEARCH_RESULT_GET_CLASS (self)->load (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_search_result_dup_icon:
 * @self: a [class@Foundry.SearchResult]
 *
 * Returns: (transfer full) (nullable):
 */
GIcon *
foundry_search_result_dup_icon (FoundrySearchResult *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SEARCH_RESULT (self), NULL);

  if (FOUNDRY_SEARCH_RESULT_GET_CLASS (self)->dup_icon)
    return FOUNDRY_SEARCH_RESULT_GET_CLASS (self)->dup_icon (self);

  return NULL;
}
