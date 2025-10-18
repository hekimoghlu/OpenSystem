/* plugin-devhelp-search-result.c
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

#include "plugin-devhelp-keyword.h"
#include "plugin-devhelp-navigatable.h"
#include "plugin-devhelp-repository.h"
#include "plugin-devhelp-search-result.h"

G_DEFINE_FINAL_TYPE (PluginDevhelpSearchResult, plugin_devhelp_search_result, FOUNDRY_TYPE_DOCUMENTATION)

enum {
  PROP_0,
  PROP_ITEM,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

PluginDevhelpSearchResult *
plugin_devhelp_search_result_new (guint position)
{
  PluginDevhelpSearchResult *self;

  self = g_object_new (PLUGIN_TYPE_DEVHELP_SEARCH_RESULT, NULL);
  self->position = position;

  return self;
}

static char *
plugin_devhelp_search_result_query_attribute (FoundryDocumentation *documentation,
                                              const char           *attribute)
{
  PluginDevhelpSearchResult *self = PLUGIN_DEVHELP_SEARCH_RESULT (documentation);

  if (FOUNDRY_IS_DOCUMENTATION (self->item))
    return foundry_documentation_query_attribute (FOUNDRY_DOCUMENTATION (self->item), attribute);

  if (PLUGIN_IS_DEVHELP_KEYWORD (self->item))
    return plugin_devhelp_keyword_query_attribute (PLUGIN_DEVHELP_KEYWORD (self->item), attribute);

  return NULL;
}

static char *
plugin_devhelp_search_result_dup_title (FoundryDocumentation *documentation)
{
  PluginDevhelpSearchResult *self = PLUGIN_DEVHELP_SEARCH_RESULT (documentation);

  if (PLUGIN_IS_DEVHELP_NAVIGATABLE (self->item))
    return g_strdup (plugin_devhelp_navigatable_get_title (PLUGIN_DEVHELP_NAVIGATABLE (self->item)));

  return NULL;
}

static char *
plugin_devhelp_search_result_dup_uri (FoundryDocumentation *documentation)
{
  PluginDevhelpSearchResult *self = PLUGIN_DEVHELP_SEARCH_RESULT (documentation);

  if (PLUGIN_IS_DEVHELP_NAVIGATABLE (self->item))
    return g_strdup (foundry_documentation_dup_uri (FOUNDRY_DOCUMENTATION (self->item)));

  return NULL;
}

static GIcon *
plugin_devhelp_search_result_dup_icon (FoundryDocumentation *documentation)
{
  PluginDevhelpSearchResult *self = PLUGIN_DEVHELP_SEARCH_RESULT (documentation);

  if (FOUNDRY_IS_DOCUMENTATION  (self->item))
    return foundry_documentation_dup_icon (FOUNDRY_DOCUMENTATION (self->item));

  return NULL;
}

static char *
plugin_devhelp_search_result_dup_section_title (FoundryDocumentation *documentation)
{
  PluginDevhelpSearchResult *self = PLUGIN_DEVHELP_SEARCH_RESULT (documentation);

  if (PLUGIN_IS_DEVHELP_NAVIGATABLE (self->item))
    return foundry_documentation_dup_section_title (FOUNDRY_DOCUMENTATION (self->item));

  return NULL;
}

static char *
plugin_devhelp_search_result_dup_deprecated_in (FoundryDocumentation *documentation)
{
  PluginDevhelpSearchResult *self = PLUGIN_DEVHELP_SEARCH_RESULT (documentation);

  if (PLUGIN_IS_DEVHELP_NAVIGATABLE (self->item))
    return foundry_documentation_dup_deprecated_in (FOUNDRY_DOCUMENTATION (self->item));

  return NULL;
}

static char *
plugin_devhelp_search_result_dup_since_version (FoundryDocumentation *documentation)
{
  PluginDevhelpSearchResult *self = PLUGIN_DEVHELP_SEARCH_RESULT (documentation);

  if (PLUGIN_IS_DEVHELP_NAVIGATABLE (self->item))
    return foundry_documentation_dup_since_version (FOUNDRY_DOCUMENTATION (self->item));

  return NULL;
}

static void
plugin_devhelp_search_result_dispose (GObject *object)
{
  PluginDevhelpSearchResult *self = (PluginDevhelpSearchResult *)object;

  if (self->model)
    plugin_devhelp_search_model_release (self->model, self);

  self->model = NULL;
  self->link.data = NULL;

  g_clear_object (&self->item);

  G_OBJECT_CLASS (plugin_devhelp_search_result_parent_class)->dispose (object);
}

static void
plugin_devhelp_search_result_get_property (GObject    *object,
                                           guint       prop_id,
                                           GValue     *value,
                                           GParamSpec *pspec)
{
  PluginDevhelpSearchResult *self = PLUGIN_DEVHELP_SEARCH_RESULT (object);

  switch (prop_id)
    {
    case PROP_ITEM:
      g_value_set_object (value, plugin_devhelp_search_result_get_item (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_devhelp_search_result_set_property (GObject      *object,
                                           guint         prop_id,
                                           const GValue *value,
                                           GParamSpec   *pspec)
{
  PluginDevhelpSearchResult *self = PLUGIN_DEVHELP_SEARCH_RESULT (object);

  switch (prop_id)
    {
    case PROP_ITEM:
      plugin_devhelp_search_result_set_item (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_devhelp_search_result_class_init (PluginDevhelpSearchResultClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDocumentationClass *documentation_class = FOUNDRY_DOCUMENTATION_CLASS (klass);

  object_class->dispose = plugin_devhelp_search_result_dispose;
  object_class->get_property = plugin_devhelp_search_result_get_property;
  object_class->set_property = plugin_devhelp_search_result_set_property;

  documentation_class->dup_icon = plugin_devhelp_search_result_dup_icon;
  documentation_class->dup_title = plugin_devhelp_search_result_dup_title;
  documentation_class->dup_uri = plugin_devhelp_search_result_dup_uri;
  documentation_class->dup_section_title = plugin_devhelp_search_result_dup_section_title;
  documentation_class->query_attribute = plugin_devhelp_search_result_query_attribute;
  documentation_class->dup_deprecated_in = plugin_devhelp_search_result_dup_deprecated_in;
  documentation_class->dup_since_version = plugin_devhelp_search_result_dup_since_version;

  properties[PROP_ITEM] =
    g_param_spec_object ("item", NULL, NULL,
                         G_TYPE_OBJECT,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_devhelp_search_result_init (PluginDevhelpSearchResult *self)
{
  self->link.data = self;
}

gpointer
plugin_devhelp_search_result_get_item (PluginDevhelpSearchResult *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_SEARCH_RESULT (self), NULL);

  return self->item;
}

void
plugin_devhelp_search_result_set_item (PluginDevhelpSearchResult *self,
                                       gpointer                   item)
{
  g_return_if_fail (PLUGIN_IS_DEVHELP_SEARCH_RESULT (self));
  g_return_if_fail (!item || G_IS_OBJECT (item));

  if (g_set_object (&self->item, item))
    {
      if (PLUGIN_IS_DEVHELP_NAVIGATABLE (item))
        {
          g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ITEM]);
          g_object_notify (G_OBJECT (self), "title");
          g_object_notify (G_OBJECT (self), "section-title");
          g_object_notify (G_OBJECT (self), "icon");

          if (PLUGIN_IS_DEVHELP_KEYWORD (plugin_devhelp_navigatable_get_item (PLUGIN_DEVHELP_NAVIGATABLE (item))))
            {
              g_object_notify (G_OBJECT (self), "since-version");
              g_object_notify (G_OBJECT (self), "deprecated-in");
            }
        }
    }
}

guint
plugin_devhelp_search_result_get_position (PluginDevhelpSearchResult *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_SEARCH_RESULT (self), 0);

  return self->position;
}
