/*
 * foundry-panel.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "foundry-panel.h"

typedef struct
{
  GtkWidget *child;
  char      *id;
  char      *title;
  GIcon     *icon;
} FoundryPanelPrivate;

enum {
  PROP_0,
  PROP_CHILD,
  PROP_ICON,
  PROP_ICON_NAME,
  PROP_ID,
  PROP_TITLE,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (FoundryPanel, foundry_panel, GTK_TYPE_WIDGET)

static GParamSpec *properties[N_PROPS];

static void
foundry_panel_dispose (GObject *object)
{
  FoundryPanel *self = (FoundryPanel *)object;
  FoundryPanelPrivate *priv = foundry_panel_get_instance_private (self);
  GtkWidget *child;

  priv->child = NULL;

  while ((child = gtk_widget_get_first_child (GTK_WIDGET (self))))
    gtk_widget_unparent (child);

  g_clear_object (&priv->icon);
  g_clear_pointer (&priv->title, g_free);
  g_clear_pointer (&priv->id, g_free);

  G_OBJECT_CLASS (foundry_panel_parent_class)->dispose (object);
}

static void
foundry_panel_get_property (GObject    *object,
                            guint       prop_id,
                            GValue     *value,
                            GParamSpec *pspec)
{
  FoundryPanel *self = FOUNDRY_PANEL (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_set_string (value, foundry_panel_get_id (self));
      break;

    case PROP_TITLE:
      g_value_set_string (value, foundry_panel_get_title (self));
      break;

    case PROP_ICON:
      g_value_set_object (value, foundry_panel_get_icon (self));
      break;

    case PROP_CHILD:
      g_value_set_object (value, foundry_panel_get_child (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_panel_set_property (GObject      *object,
                            guint         prop_id,
                            const GValue *value,
                            GParamSpec   *pspec)
{
  FoundryPanel *self = FOUNDRY_PANEL (object);
  FoundryPanelPrivate *priv = foundry_panel_get_instance_private (self);

  switch (prop_id)
    {
    case PROP_ID:
      priv->id = g_value_dup_string (value);
      break;

    case PROP_ICON:
      foundry_panel_set_icon (self, g_value_get_object (value));
      break;

    case PROP_ICON_NAME:
      foundry_panel_set_icon_name (self, g_value_get_string (value));
      break;

    case PROP_TITLE:
      foundry_panel_set_title (self, g_value_get_string (value));
      break;

    case PROP_CHILD:
      foundry_panel_set_child (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_panel_class_init (FoundryPanelClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->dispose = foundry_panel_dispose;
  object_class->get_property = foundry_panel_get_property;
  object_class->set_property = foundry_panel_set_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_CHILD] =
    g_param_spec_object ("child", NULL, NULL,
                         GTK_TYPE_WIDGET,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ICON] =
    g_param_spec_object ("icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ICON_NAME] =
    g_param_spec_string ("icon-name", NULL, NULL,
                         NULL,
                         (G_PARAM_WRITABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  gtk_widget_class_set_layout_manager_type (widget_class, GTK_TYPE_BIN_LAYOUT);
}

static void
foundry_panel_init (FoundryPanel *self)
{
}

FoundryPanel *
foundry_panel_new (const char *id)
{
  g_return_val_if_fail (id != NULL, NULL);

  return g_object_new (FOUNDRY_TYPE_PANEL,
                       "id", id,
                       NULL);
}

const char *
foundry_panel_get_id (FoundryPanel *self)
{
  FoundryPanelPrivate *priv = foundry_panel_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_PANEL (self), NULL);

  return priv->id;
}

/**
 * foundry_panel_get_title:
 * @self: a [class@FoundryAdw.Panel]
 *
 * Returns: (nullable):
 */
const char *
foundry_panel_get_title (FoundryPanel *self)
{
  FoundryPanelPrivate *priv = foundry_panel_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_PANEL (self), NULL);

  return priv->title;
}

void
foundry_panel_set_title (FoundryPanel *self,
                         const char   *title)
{
  FoundryPanelPrivate *priv = foundry_panel_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_PANEL (self));

  if (g_set_str (&priv->title, title))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_TITLE]);
}

/**
 * foundry_panel_get_icon:
 * @self: a [class@FoundryAdw.Panel]
 *
 * Returns: (transfer none) (nullable):
 */
GIcon *
foundry_panel_get_icon (FoundryPanel *self)
{
  FoundryPanelPrivate *priv = foundry_panel_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_PANEL (self), NULL);

  return priv->icon;
}

void
foundry_panel_set_icon (FoundryPanel *self,
                        GIcon        *icon)
{
  FoundryPanelPrivate *priv = foundry_panel_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_PANEL (self));

  if (g_set_object (&priv->icon, icon))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ICON]);
}

void
foundry_panel_set_icon_name (FoundryPanel *self,
                             const char   *icon_name)
{
  FoundryPanelPrivate *priv = foundry_panel_get_instance_private (self);
  g_autoptr(GIcon) icon = NULL;

  g_return_if_fail (FOUNDRY_IS_PANEL (self));

  if (icon_name != NULL)
    icon = g_themed_icon_new (icon_name);

  if (g_set_object (&priv->icon, icon))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ICON]);
}

/**
 * foundry_panel_get_child:
 * @self: a [class@FoundryAdw.Panel]
 *
 * Returns: (transfer none) (nullable):
 */
GtkWidget *
foundry_panel_get_child (FoundryPanel *self)
{
  FoundryPanelPrivate *priv = foundry_panel_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_PANEL (self), NULL);

  return priv->child;
}

void
foundry_panel_set_child (FoundryPanel *self,
                         GtkWidget    *child)
{
  FoundryPanelPrivate *priv = foundry_panel_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_PANEL (self));
  g_return_if_fail (!child || GTK_IS_WIDGET (child));

  if (child != NULL)
    gtk_widget_set_parent (child, GTK_WIDGET (self));

  g_clear_pointer (&priv->child, gtk_widget_unparent);
  priv->child = child;

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CHILD]);
}
