/*
 * manuals-search-row.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "config.h"

#include "manuals-search-row.h"

struct _ManualsSearchRow
{
  GtkWidget parent_instance;
  guint warning : 1;
};

enum {
  PROP_0,
  PROP_WARNING,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (ManualsSearchRow, manuals_search_row, GTK_TYPE_WIDGET)

static GParamSpec *properties[N_PROPS];

static void
manuals_search_row_root (GtkWidget *widget)
{
  ManualsSearchRow *self = MANUALS_SEARCH_ROW (widget);
  GtkWidget *parent;

  GTK_WIDGET_CLASS (manuals_search_row_parent_class)->root (widget);

  parent = gtk_widget_get_parent (widget);

  if (self->warning)
    gtk_widget_add_css_class (parent, "warning");
}

static void
manuals_search_row_dispose (GObject *object)
{
  GtkWidget *child;

  while ((child = gtk_widget_get_first_child (GTK_WIDGET (object))))
    gtk_widget_unparent (child);

  G_OBJECT_CLASS (manuals_search_row_parent_class)->dispose (object);
}

static void
manuals_search_row_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  ManualsSearchRow *self = MANUALS_SEARCH_ROW (object);

  switch (prop_id)
    {
    case PROP_WARNING:
      g_value_set_boolean (value, manuals_search_row_get_warning (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
manuals_search_row_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  ManualsSearchRow *self = MANUALS_SEARCH_ROW (object);

  switch (prop_id)
    {
    case PROP_WARNING:
      manuals_search_row_set_warning (self, g_value_get_boolean (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
manuals_search_row_class_init (ManualsSearchRowClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->dispose = manuals_search_row_dispose;
  object_class->get_property = manuals_search_row_get_property;
  object_class->set_property = manuals_search_row_set_property;

  widget_class->root = manuals_search_row_root;

  properties[PROP_WARNING] =
    g_param_spec_boolean ("warning", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  gtk_widget_class_set_layout_manager_type (widget_class, GTK_TYPE_BIN_LAYOUT);
}

static void
manuals_search_row_init (ManualsSearchRow *self)
{
}

gboolean
manuals_search_row_get_warning (ManualsSearchRow *self)
{
  g_return_val_if_fail (MANUALS_IS_SEARCH_ROW (self), FALSE);

  return self->warning;
}

void
manuals_search_row_set_warning (ManualsSearchRow *self,
                                gboolean          warning)
{
  g_return_if_fail (MANUALS_IS_SEARCH_ROW (self));

  warning = !!warning;

  if (warning != self->warning)
    {
      GtkWidget *widget = gtk_widget_get_parent (GTK_WIDGET (self));

      self->warning = warning;

      if (warning && widget)
        gtk_widget_add_css_class (widget, "warning");
      else
        gtk_widget_remove_css_class (widget, "warning");

      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_WARNING]);
    }
}
