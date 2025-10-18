/* foundry-panel-button.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "foundry-panel-button-private.h"

struct _FoundryPanelButton
{
  GtkWidget parent_instance;
  FoundryWorkspaceChild *panel;
};

G_DEFINE_FINAL_TYPE (FoundryPanelButton, foundry_panel_button, GTK_TYPE_WIDGET)

enum {
  PROP_0,
  PROP_PANEL,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_panel_button_dispose (GObject *object)
{
  FoundryPanelButton *self = (FoundryPanelButton *)object;
  GtkWidget *child;

  gtk_widget_dispose_template (GTK_WIDGET (object), FOUNDRY_TYPE_PANEL_BUTTON);

  while ((child = gtk_widget_get_first_child (GTK_WIDGET (self))))
    gtk_widget_unparent (child);

  g_clear_object (&self->panel);

  G_OBJECT_CLASS (foundry_panel_button_parent_class)->dispose (object);
}

static void
foundry_panel_button_get_property (GObject    *object,
                                   guint       prop_id,
                                   GValue     *value,
                                   GParamSpec *pspec)
{
  FoundryPanelButton *self = FOUNDRY_PANEL_BUTTON (object);

  switch (prop_id)
    {
    case PROP_PANEL:
      g_value_set_object (value, self->panel);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_panel_button_set_property (GObject      *object,
                                   guint         prop_id,
                                   const GValue *value,
                                   GParamSpec   *pspec)
{
  FoundryPanelButton *self = FOUNDRY_PANEL_BUTTON (object);

  switch (prop_id)
    {
    case PROP_PANEL:
      self->panel = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_panel_button_class_init (FoundryPanelButtonClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->dispose = foundry_panel_button_dispose;
  object_class->get_property = foundry_panel_button_get_property;
  object_class->set_property = foundry_panel_button_set_property;

  properties[PROP_PANEL] =
    g_param_spec_object ("panel", NULL, NULL,
                         FOUNDRY_TYPE_WORKSPACE_CHILD,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  gtk_widget_class_set_template_from_resource (widget_class, "/app/devsuite/foundry-adw/ui/foundry-panel-button.ui");
  gtk_widget_class_set_layout_manager_type (widget_class, GTK_TYPE_BIN_LAYOUT);
}

static void
foundry_panel_button_init (FoundryPanelButton *self)
{
  gtk_widget_init_template (GTK_WIDGET (self));
}

GtkWidget *
foundry_panel_button_new (FoundryWorkspaceChild *panel)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (panel), NULL);

  return g_object_new (FOUNDRY_TYPE_PANEL_BUTTON,
                       "panel", panel,
                       NULL);
}
