/* foundry-documentation-matches.c
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

#include "foundry-documentation.h"
#include "foundry-documentation-query.h"
#include "foundry-documentation-matches-private.h"
#include "foundry-model-manager.h"

struct _FoundryDocumentationMatches
{
  GObject                    parent_instance;
  FoundryDocumentationQuery *query;
  GListStore                *sections;
  GListModel                *flatten;
  DexFuture                 *future;
};

enum {
  PROP_0,
  PROP_QUERY,
  PROP_SECTIONS,
  N_PROPS
};

static GType
foundry_documentation_matches_get_item_type (GListModel *model)
{
  return FOUNDRY_TYPE_DOCUMENTATION;
}

static guint
foundry_documentation_matches_get_n_items (GListModel *model)
{
  return g_list_model_get_n_items (FOUNDRY_DOCUMENTATION_MATCHES (model)->flatten);
}

static gpointer
foundry_documentation_matches_get_item (GListModel *model,
                                        guint       position)
{
  return g_list_model_get_item (FOUNDRY_DOCUMENTATION_MATCHES (model)->flatten, position);
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_item_type = foundry_documentation_matches_get_item_type;
  iface->get_n_items = foundry_documentation_matches_get_n_items;
  iface->get_item = foundry_documentation_matches_get_item;
}

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryDocumentationMatches, foundry_documentation_matches, G_TYPE_OBJECT,
                               G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

static GParamSpec *properties[N_PROPS];

static void
foundry_documentation_matches_dispose (GObject *object)
{
  FoundryDocumentationMatches *self = (FoundryDocumentationMatches *)object;

  g_clear_object (&self->query);
  g_clear_object (&self->flatten);
  g_clear_object (&self->sections);
  dex_clear (&self->future);

  G_OBJECT_CLASS (foundry_documentation_matches_parent_class)->dispose (object);
}

static void
foundry_documentation_matches_get_property (GObject    *object,
                                            guint       prop_id,
                                            GValue     *value,
                                            GParamSpec *pspec)
{
  FoundryDocumentationMatches *self = FOUNDRY_DOCUMENTATION_MATCHES (object);

  switch (prop_id)
    {
    case PROP_QUERY:
      g_value_take_object (value, foundry_documentation_matches_dup_query (self));
      break;

    case PROP_SECTIONS:
      g_value_take_object (value, foundry_documentation_matches_list_sections (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_documentation_matches_set_property (GObject      *object,
                                            guint         prop_id,
                                            const GValue *value,
                                            GParamSpec   *pspec)
{
  FoundryDocumentationMatches *self = FOUNDRY_DOCUMENTATION_MATCHES (object);

  switch (prop_id)
    {
    case PROP_QUERY:
      self->query = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_documentation_matches_class_init (FoundryDocumentationMatchesClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_documentation_matches_dispose;
  object_class->get_property = foundry_documentation_matches_get_property;
  object_class->set_property = foundry_documentation_matches_set_property;

  properties[PROP_QUERY] =
    g_param_spec_object ("query", NULL, NULL,
                         FOUNDRY_TYPE_DOCUMENTATION_QUERY,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SECTIONS] =
    g_param_spec_object ("sections", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_documentation_matches_init (FoundryDocumentationMatches *self)
{
  self->sections = g_list_store_new (G_TYPE_LIST_MODEL);
  self->flatten = foundry_flatten_list_model_new (g_object_ref (G_LIST_MODEL (self->sections)));
}

FoundryDocumentationMatches *
foundry_documentation_matches_new (FoundryDocumentationQuery *query)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_QUERY (query), NULL);

  return g_object_new (FOUNDRY_TYPE_DOCUMENTATION_MATCHES,
                       "query", query,
                       NULL);
}

void
foundry_documentation_matches_add_section (FoundryDocumentationMatches *self,
                                           GListModel                  *section)
{
  g_return_if_fail (FOUNDRY_IS_DOCUMENTATION_MATCHES (self));
  g_return_if_fail (G_IS_LIST_MODEL (section));

  g_list_store_append (self->sections, section);
}

/**
 * foundry_documentation_matches_list_sections:
 * @self: a [class@Foundry.DocumentationMatches]
 *
 * Returns: (transfer full): an [iface@Gio.ListModel] of [class@Foundry.Documentation]
 */
GListModel *
foundry_documentation_matches_list_sections (FoundryDocumentationMatches *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_MATCHES (self), NULL);

  return g_object_ref (G_LIST_MODEL (self->sections));
}

/**
 * foundry_documentation_matches_dup_query:
 * @self: a [class@Foundry.DocumentationMatches]
 *
 * Returns: (transfer full):
 */
FoundryDocumentationQuery *
foundry_documentation_matches_dup_query (FoundryDocumentationMatches *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DOCUMENTATION_MATCHES (self), NULL);

  return g_object_ref (self->query);
}

void
foundry_documentation_matches_set_future (FoundryDocumentationMatches *self,
                                          DexFuture                   *future)
{
  g_return_if_fail (FOUNDRY_IS_DOCUMENTATION_MATCHES (self));
  g_return_if_fail (!future || DEX_IS_FUTURE (future));

  if (future)
    dex_ref (future);
  dex_clear (&self->future);
  self->future = future;
}

/**
 * foundry_documentation_matches_await:
 * @self: a [class@Foundry.DocumentationMatches]
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any value
 *   when populating the list model has completed or rejects with error.
 */
DexFuture *
foundry_documentation_matches_await (FoundryDocumentationMatches *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DOCUMENTATION_MATCHES (self));

  if (self->future != NULL)
    return dex_ref (self->future);

  return dex_future_new_true ();
}
