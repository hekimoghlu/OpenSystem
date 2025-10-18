/* plugin-devhelp-heading.c
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

#include "plugin-devhelp-book.h"
#include "plugin-devhelp-heading.h"
#include "plugin-devhelp-navigatable.h"
#include "plugin-devhelp-repository.h"

struct _PluginDevhelpHeading
{
  GomResource parent_instance;
  gint64 id;
  gint64 parent_id;
  gint64 book_id;
  char *title;
  char *uri;
  guint has_children : 1;
};

G_DEFINE_FINAL_TYPE (PluginDevhelpHeading, plugin_devhelp_heading, GOM_TYPE_RESOURCE)

enum {
  PROP_0,
  PROP_HAS_CHILDREN,
  PROP_ID,
  PROP_PARENT_ID,
  PROP_BOOK_ID,
  PROP_TITLE,
  PROP_URI,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

static void
plugin_devhelp_heading_finalize (GObject *object)
{
  PluginDevhelpHeading *self = (PluginDevhelpHeading *)object;

  g_clear_pointer (&self->title, g_free);
  g_clear_pointer (&self->uri, g_free);

  G_OBJECT_CLASS (plugin_devhelp_heading_parent_class)->finalize (object);
}

static void
plugin_devhelp_heading_get_property (GObject    *object,
                              guint       prop_id,
                              GValue     *value,
                              GParamSpec *pspec)
{
  PluginDevhelpHeading *self = PLUGIN_DEVHELP_HEADING (object);

  switch (prop_id)
    {
    case PROP_HAS_CHILDREN:
      g_value_set_boolean (value, self->has_children);
      break;

    case PROP_ID:
      g_value_set_int64 (value, plugin_devhelp_heading_get_id (self));
      break;

    case PROP_PARENT_ID:
      g_value_set_int64 (value, plugin_devhelp_heading_get_parent_id (self));
      break;

    case PROP_BOOK_ID:
      g_value_set_int64 (value, plugin_devhelp_heading_get_book_id (self));
      break;

    case PROP_TITLE:
      g_value_set_string (value, plugin_devhelp_heading_get_title (self));
      break;

    case PROP_URI:
      g_value_set_string (value, plugin_devhelp_heading_get_uri (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_devhelp_heading_set_property (GObject      *object,
                              guint         prop_id,
                              const GValue *value,
                              GParamSpec   *pspec)
{
  PluginDevhelpHeading *self = PLUGIN_DEVHELP_HEADING (object);

  switch (prop_id)
    {
    case PROP_HAS_CHILDREN:
      self->has_children = g_value_get_boolean (value);
      break;

    case PROP_ID:
      plugin_devhelp_heading_set_id (self, g_value_get_int64 (value));
      break;

    case PROP_PARENT_ID:
      plugin_devhelp_heading_set_parent_id (self, g_value_get_int64 (value));
      break;

    case PROP_BOOK_ID:
      plugin_devhelp_heading_set_book_id (self, g_value_get_int64 (value));
      break;

    case PROP_TITLE:
      plugin_devhelp_heading_set_title (self, g_value_get_string (value));
      break;

    case PROP_URI:
      plugin_devhelp_heading_set_uri (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_devhelp_heading_class_init (PluginDevhelpHeadingClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GomResourceClass *resource_class = GOM_RESOURCE_CLASS (klass);

  object_class->finalize = plugin_devhelp_heading_finalize;
  object_class->get_property = plugin_devhelp_heading_get_property;
  object_class->set_property = plugin_devhelp_heading_set_property;

  properties[PROP_HAS_CHILDREN] =
    g_param_spec_boolean ("has-children", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_ID] =
    g_param_spec_int64 ("id", NULL, NULL,
                        0, G_MAXINT64, 0,
                        (G_PARAM_READWRITE |
                         G_PARAM_EXPLICIT_NOTIFY |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_PARENT_ID] =
    g_param_spec_int64 ("parent-id", NULL, NULL,
                        0, G_MAXINT64, 0,
                        (G_PARAM_READWRITE |
                         G_PARAM_EXPLICIT_NOTIFY |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_BOOK_ID] =
    g_param_spec_int64 ("book-id", NULL, NULL,
                        0, G_MAXINT64, 0,
                        (G_PARAM_READWRITE |
                         G_PARAM_EXPLICIT_NOTIFY |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
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

  gom_resource_class_set_table (resource_class, "headings");
  gom_resource_class_set_primary_key (resource_class, "id");
  gom_resource_class_set_notnull (resource_class, "title");
  gom_resource_class_set_reference (resource_class, "parent-id", "headings", "id");
  gom_resource_class_set_reference (resource_class, "book-id", "books", "id");
}

static void
plugin_devhelp_heading_init (PluginDevhelpHeading *self)
{
}

const char *
plugin_devhelp_heading_get_title (PluginDevhelpHeading *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_HEADING (self), NULL);

  return self->title;
}

void
plugin_devhelp_heading_set_title (PluginDevhelpHeading *self,
                                  const char     *title)
{
  g_return_if_fail (PLUGIN_IS_DEVHELP_HEADING (self));

  if (g_set_str (&self->title, title))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_TITLE]);
}

const char *
plugin_devhelp_heading_get_uri (PluginDevhelpHeading *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_HEADING (self), NULL);

  return self->uri;
}

void
plugin_devhelp_heading_set_uri (PluginDevhelpHeading *self,
                                const char     *uri)
{
  g_return_if_fail (PLUGIN_IS_DEVHELP_HEADING (self));

  if (g_set_str (&self->uri, uri))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_URI]);
}

gint64
plugin_devhelp_heading_get_id (PluginDevhelpHeading *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_HEADING (self), 0);

  return self->id;
}

void
plugin_devhelp_heading_set_id (PluginDevhelpHeading *self,
                               gint64          id)
{
  g_return_if_fail (PLUGIN_IS_DEVHELP_HEADING (self));
  g_return_if_fail (id >= 0);

  if (self->id != id)
    {
      self->id = id;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ID]);
    }
}

gint64
plugin_devhelp_heading_get_parent_id (PluginDevhelpHeading *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_HEADING (self), 0);

  return self->parent_id;
}

void
plugin_devhelp_heading_set_parent_id (PluginDevhelpHeading *self,
                                      gint64          parent_id)
{
  g_return_if_fail (PLUGIN_IS_DEVHELP_HEADING (self));
  g_return_if_fail (parent_id >= 0);

  if (self->parent_id != parent_id)
    {
      self->parent_id = parent_id;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_PARENT_ID]);
    }
}

gint64
plugin_devhelp_heading_get_book_id (PluginDevhelpHeading *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_HEADING (self), 0);

  return self->book_id;
}

void
plugin_devhelp_heading_set_book_id (PluginDevhelpHeading *self,
                                    gint64          book_id)
{
  g_return_if_fail (PLUGIN_IS_DEVHELP_HEADING (self));
  g_return_if_fail (book_id >= 0);

  if (self->book_id != book_id)
    {
      self->book_id = book_id;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_BOOK_ID]);
    }
}

DexFuture *
plugin_devhelp_heading_find_parent (PluginDevhelpHeading *self)
{
  g_autoptr(GomFilter) filter = NULL;
  g_autoptr(PluginDevhelpRepository) repository = NULL;
  GValue parent_id = G_VALUE_INIT;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_HEADING (self), NULL);

  g_object_get (self,
                "repository", &repository,
                NULL);

  if (repository == NULL || self->parent_id <= 0)
    return plugin_devhelp_heading_find_book (self);

  g_value_init (&parent_id, G_TYPE_INT64);
  g_value_set_int64 (&parent_id, self->parent_id);
  filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_HEADING, "id", &parent_id);
  g_value_unset (&parent_id);

  return plugin_devhelp_repository_find_one (repository, PLUGIN_TYPE_DEVHELP_HEADING, filter);
}

DexFuture *
plugin_devhelp_heading_find_sdk (PluginDevhelpHeading *self)
{
  g_autoptr(GomFilter) filter = NULL;
  g_autoptr(PluginDevhelpRepository) repository = NULL;
  GValue book_id = G_VALUE_INIT;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_HEADING (self), NULL);

  g_object_get (self,
                "repository", &repository,
                NULL);

  if (repository == NULL || self->book_id <= 0)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_FOUND,
                                  "Failed to locate SDK");

  g_value_init (&book_id, G_TYPE_INT64);
  g_value_set_int64 (&book_id, self->book_id);
  filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_SDK, "id", &book_id);
  g_value_unset (&book_id);

  return plugin_devhelp_repository_find_one (repository, PLUGIN_TYPE_DEVHELP_SDK, filter);
}

DexFuture *
plugin_devhelp_heading_list_headings (PluginDevhelpHeading *self)
{
  g_autoptr(PluginDevhelpRepository) repository = NULL;
  g_autoptr(GomFilter) filter = NULL;
  g_auto(GValue) value = G_VALUE_INIT;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_HEADING (self), NULL);

  g_object_get (self, "repository", &repository, NULL);

  if (repository == NULL)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_SUPPORTED,
                                  "No repository to query");

  g_value_init (&value, G_TYPE_INT64);
  g_value_set_int64 (&value, self->id);
  filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_HEADING, "parent-id", &value);

  return plugin_devhelp_repository_list (repository, PLUGIN_TYPE_DEVHELP_HEADING, filter);
}

DexFuture *
plugin_devhelp_heading_find_book (PluginDevhelpHeading *self)
{
  g_autoptr(PluginDevhelpRepository) repository = NULL;
  g_autoptr(GomFilter) filter = NULL;
  g_auto(GValue) value = G_VALUE_INIT;

  g_return_val_if_fail (PLUGIN_IS_DEVHELP_HEADING (self), NULL);

  g_object_get (self, "repository", &repository, NULL);

  if (repository == NULL)
    return dex_future_new_reject (G_IO_ERROR,
                                  G_IO_ERROR_NOT_SUPPORTED,
                                  "No repository to query");

  g_value_init (&value, G_TYPE_INT64);
  g_value_set_int64 (&value, self->book_id);
  filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_BOOK, "id", &value);

  return plugin_devhelp_repository_find_one (repository, PLUGIN_TYPE_DEVHELP_BOOK, filter);
}

static GomFilter *
create_prefix_filter (const char *uri)
{
  g_autoptr(GArray) values = NULL;
  g_autoptr(GomFilter) filter = NULL;
  GValue *value;

  values = g_array_new (FALSE, TRUE, sizeof (GValue));
  g_array_set_clear_func (values, (GDestroyNotify)g_value_unset);
  g_array_set_size (values, 1);

  value = &g_array_index (values, GValue, 0);
  g_value_init (value, G_TYPE_STRING);
  g_value_set_string (value, uri);

  return gom_filter_new_sql ("? LIKE replace(\"online-uri\",'%','') || '%'", values);
}

static char *
get_parent_uri (const char *uri)
{
  const char *end;

  if (g_str_has_suffix (uri, "/"))
    return g_strdup (uri);

  if (!(end = strrchr (uri, '/')))
    return NULL;

  return g_strndup (uri, end - uri + 1);
}

static DexFuture *
plugin_devhelp_heading_find_by_uri_fiber (PluginDevhelpRepository *repository,
                                          const char              *uri)
{
  g_autoptr(PluginDevhelpHeading) heading = NULL;
  g_autoptr(PluginDevhelpBook) book = NULL;
  g_autoptr(GomFilter) filter = NULL;
  g_autoptr(GomFilter) prefix_filter = NULL;
  g_auto(GValue) value = G_VALUE_INIT;

  g_assert (PLUGIN_IS_DEVHELP_REPOSITORY (repository));
  g_assert (uri != NULL);

  g_value_init (&value, G_TYPE_STRING);
  g_value_set_string (&value, uri);
  filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_HEADING, "uri", &value);

  if ((heading = dex_await_object (plugin_devhelp_repository_find_one (repository, PLUGIN_TYPE_DEVHELP_HEADING, filter), NULL)))
    return dex_future_new_take_object (g_steal_pointer (&heading));

  g_clear_object (&filter);

  /* Okay we didn't find anything with exact match. Next try to find a book
   * that has a "online-uri" which is a prefix of @uri.
   */
  prefix_filter = create_prefix_filter (uri);
  if ((book = dex_await_object (plugin_devhelp_repository_find_one (repository, PLUGIN_TYPE_DEVHELP_BOOK, prefix_filter), NULL)))
    {
      const char *default_uri = plugin_devhelp_book_get_default_uri (book);
      const char *online_uri = plugin_devhelp_book_get_online_uri (book);

      /* Now try find a heading that has the same suffix as what
       * @uri is beyond the book's online-uri.
       */

      if (online_uri != NULL &&
          default_uri != NULL &&
          g_str_has_prefix (uri, online_uri))
        {
          const char *suffix = uri + strlen (online_uri);
          g_autofree char *parent = NULL;
          g_autofree char *alternate = NULL;

          /* We don't currently store the root directory of the book and
           * instead store a default-uri. In every case I've seen the default-uri
           * is in the immediate child of that directory, so just use that
           * convention here to make our guess.
           */
          parent = get_parent_uri (default_uri);
          while (suffix[0] == '/')
            suffix++;
          alternate = g_strconcat (parent, suffix, NULL);

          g_value_set_string (&value, alternate);
          filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_HEADING, "uri", &value);

          if ((heading = dex_await_object (plugin_devhelp_repository_find_one (repository, PLUGIN_TYPE_DEVHELP_HEADING, filter), NULL)))
            return dex_future_new_take_object (g_steal_pointer (&heading));
        }
    }

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_FOUND,
                                "Not found");
}

DexFuture *
plugin_devhelp_heading_find_by_uri (PluginDevhelpRepository *repository,
                                    const char              *uri)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_REPOSITORY (repository), NULL);
  g_return_val_if_fail (uri != NULL, NULL);

  return foundry_scheduler_spawn (NULL, 0,
                                  G_CALLBACK (plugin_devhelp_heading_find_by_uri_fiber),
                                  2,
                                  PLUGIN_TYPE_DEVHELP_REPOSITORY, repository,
                                  G_TYPE_STRING, uri);
}

static DexFuture *
plugin_devhelp_heading_list_alternates_fiber (gpointer data)
{
  PluginDevhelpHeading *self = data;
  g_autoptr(PluginDevhelpRepository) repository = NULL;
  g_autoptr(GomFilter) books_filter = NULL;
  g_autoptr(GomFilter) heading_filter = NULL;
  g_autoptr(GListStore) store = NULL;
  g_autoptr(GListModel) books = NULL;
  g_autoptr(PluginDevhelpBook) book = NULL;
  g_auto(GValue) title_value = G_VALUE_INIT;
  g_auto(GValue) heading_value = G_VALUE_INIT;
  guint n_books;

  g_assert (PLUGIN_IS_DEVHELP_HEADING (self));

  store = g_list_store_new (PLUGIN_TYPE_DEVHELP_NAVIGATABLE);
  g_object_get (self, "repository", &repository, NULL);
  if (repository == NULL)
    goto failure;

  /* First find the book for this heading */
  if (!(book = dex_await_object (plugin_devhelp_heading_find_book (self), NULL)))
    goto failure;

  /* Now create filter where book title is same */
  g_value_init (&title_value, G_TYPE_STRING);
  g_object_get_property (G_OBJECT (book), "title", &title_value);
  books_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_BOOK, "title", &title_value);

  /* Now find other books that have the same title */
  if (!(books = dex_await_object (plugin_devhelp_repository_list (repository,
                                                           PLUGIN_TYPE_DEVHELP_BOOK,
                                                           books_filter),
                                  NULL)))
    goto failure;

  /* Look for matching headings of each book */
  g_value_init (&heading_value, G_TYPE_STRING);
  g_value_set_string (&heading_value, self->title);
  heading_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_HEADING, "title", &heading_value);
  n_books = g_list_model_get_n_items (books);
  for (guint i = 0; i < n_books; i++)
    {
      g_autoptr(PluginDevhelpBook) this_book = g_list_model_get_item (books, i);
      g_autoptr(PluginDevhelpNavigatable) navigatable = NULL;
      g_autoptr(PluginDevhelpHeading) match = NULL;
      g_autoptr(PluginDevhelpSdk) sdk = NULL;
      g_autoptr(GomFilter) book_id_filter = NULL;
      g_autoptr(GomFilter) filter = NULL;
      g_autoptr(GomFilter) sdk_filter = NULL;
      g_auto(GValue) book_id_value = G_VALUE_INIT;
      g_auto(GValue) sdk_id_value = G_VALUE_INIT;
      g_autofree char *title = NULL;
      g_autofree char *sdk_title = NULL;
      g_autoptr(GIcon) jump_icon = NULL;
      const char *icon_name;
      gint64 sdk_id;

      if (plugin_devhelp_book_get_id (this_book) == self->book_id)
        continue;

      g_value_init (&book_id_value, G_TYPE_INT64);
      g_value_set_int64 (&book_id_value, plugin_devhelp_book_get_id (this_book));

      book_id_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_HEADING, "book-id", &book_id_value);
      filter = gom_filter_new_and (book_id_filter, heading_filter);

      /* Find the matching heading for this book */
      if (!(match = dex_await_object (plugin_devhelp_repository_find_one (repository,
                                                                   PLUGIN_TYPE_DEVHELP_HEADING,
                                                                   filter),
                                      NULL)))
        continue;

      sdk_id = plugin_devhelp_repository_get_cached_sdk_id (repository, plugin_devhelp_book_get_id (this_book));
      g_value_init (&sdk_id_value, G_TYPE_INT64);
      g_value_set_int64 (&sdk_id_value, sdk_id);
      sdk_filter = gom_filter_new_eq (PLUGIN_TYPE_DEVHELP_SDK, "id", &sdk_id_value);

      /* Get the SDK title for this book */
      if (!(sdk = dex_await_object (plugin_devhelp_repository_find_one (repository,
                                                                 PLUGIN_TYPE_DEVHELP_SDK,
                                                                 sdk_filter),
                                    NULL)))
        continue;

      if ((icon_name = plugin_devhelp_sdk_get_icon_name (sdk)))
        jump_icon = g_themed_icon_new (icon_name);

      sdk_title = plugin_devhelp_sdk_dup_title (sdk);
      title = g_strdup_printf (_("View in %s"), sdk_title);
      navigatable = plugin_devhelp_navigatable_new_for_resource (G_OBJECT (match));
      g_object_set (navigatable,
                    "menu-title", title,
                    "menu-icon", jump_icon,
                    NULL);

      g_list_store_append (store, navigatable);
    }

failure:
  return dex_future_new_take_object (g_steal_pointer (&store));
}

DexFuture *
plugin_devhelp_heading_list_alternates (PluginDevhelpHeading *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_HEADING (self), NULL);

  return dex_scheduler_spawn (NULL, 0,
                              plugin_devhelp_heading_list_alternates_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

gboolean
plugin_devhelp_heading_has_children (PluginDevhelpHeading *self)
{
  g_return_val_if_fail (PLUGIN_IS_DEVHELP_HEADING (self), FALSE);

  return self->has_children;
}
