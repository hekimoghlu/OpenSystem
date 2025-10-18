/* foundry-panel-bar.c
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

#include "foundry-panel-bar.h"
#include "foundry-panel-button-private.h"
#include "foundry-workspace-private.h"
#include "foundry-workspace-child-private.h"

struct _FoundryPanelBar
{
  GtkWidget           parent_instance;

  FoundryWorkspace   *workspace;
  GtkFilterListModel *panels;
  GtkCustomFilter    *filter;

  guint               show_start : 1;
  guint               show_bottom : 1;
};

G_DEFINE_FINAL_TYPE (FoundryPanelBar, foundry_panel_bar, GTK_TYPE_WIDGET)

enum {
  PROP_0,
  PROP_WORKSPACE,
  PROP_SHOW_START,
  PROP_SHOW_BOTTOM,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static GtkWidget *
foundry_panel_bar_get_nth (FoundryPanelBar *self,
                           guint            nth)
{
  GtkWidget *child;

  g_assert (FOUNDRY_IS_PANEL_BAR (self));

  child = gtk_widget_get_first_child (GTK_WIDGET (self));

  while (child && nth)
    {
      child = gtk_widget_get_next_sibling (child);
      nth--;
    }

  return child;
}

static void
foundry_panel_bar_items_changed_cb (FoundryPanelBar *self,
                                    guint            position,
                                    guint            removed,
                                    guint            added,
                                    GListModel      *model)
{
  GtkWidget *child = NULL;

  g_assert (FOUNDRY_IS_PANEL_BAR (self));
  g_assert (G_IS_LIST_MODEL (model));

  child = foundry_panel_bar_get_nth (self, position);

  for (guint i = 0; i < removed; i++)
    {
      GtkWidget *tmp = gtk_widget_get_next_sibling (child);
      gtk_widget_unparent (child);
      child = tmp;
    }

  if (added == 0)
    return;

  for (guint i = 0; i < added; i++)
    {
      g_autoptr(FoundryWorkspaceChild) item = g_list_model_get_item (model, position + i);
      GtkWidget *button = foundry_panel_button_new (item);

      gtk_widget_insert_before (button, GTK_WIDGET (self), child);
    }
}

static gboolean
filter_func (gpointer item,
             gpointer user_data)
{
  FoundryPanelBar *self = FOUNDRY_PANEL_BAR (user_data);
  FoundryWorkspaceChild *child = FOUNDRY_WORKSPACE_CHILD (item);
  FoundryWorkspaceChildKind kind = foundry_workspace_child_get_kind (child);

  if (kind != FOUNDRY_WORKSPACE_CHILD_PANEL)
    return FALSE;

  switch (foundry_workspace_child_get_area (child))
    {
      case PANEL_AREA_START:
        return self->show_start;

      case PANEL_AREA_BOTTOM:
        return self->show_bottom;

      case PANEL_AREA_END:
      case PANEL_AREA_TOP:
      case PANEL_AREA_CENTER:
      default:
        return FALSE;
    }
}

static void
foundry_panel_bar_dispose (GObject *object)
{
  FoundryPanelBar *self = (FoundryPanelBar *)object;
  GtkWidget *child;

  if (self->panels != NULL)
    {
      gtk_filter_list_model_set_model (self->panels, NULL);
      gtk_filter_list_model_set_filter (self->panels, NULL);
    }

  if (self->filter != NULL)
    gtk_custom_filter_set_filter_func (self->filter, NULL, NULL, NULL);

  gtk_widget_dispose_template (GTK_WIDGET (object), FOUNDRY_TYPE_PANEL_BAR);

  while ((child = gtk_widget_get_first_child (GTK_WIDGET (self))))
    gtk_widget_unparent (child);

  g_clear_weak_pointer (&self->workspace);

  G_OBJECT_CLASS (foundry_panel_bar_parent_class)->dispose (object);
}

static void
foundry_panel_bar_finalize (GObject *object)
{
  FoundryPanelBar *self = (FoundryPanelBar *)object;

  g_clear_object (&self->panels);
  g_clear_object (&self->filter);

  G_OBJECT_CLASS (foundry_panel_bar_parent_class)->finalize (object);
}

static void
foundry_panel_bar_get_property (GObject    *object,
                                guint       prop_id,
                                GValue     *value,
                                GParamSpec *pspec)
{
  FoundryPanelBar *self = FOUNDRY_PANEL_BAR (object);

  switch (prop_id)
    {
    case PROP_SHOW_BOTTOM:
      g_value_set_boolean (value, foundry_panel_bar_get_show_bottom (self));
      break;

    case PROP_SHOW_START:
      g_value_set_boolean (value, foundry_panel_bar_get_show_start (self));
      break;

    case PROP_WORKSPACE:
      g_value_set_object (value, foundry_panel_bar_get_workspace (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_panel_bar_set_property (GObject      *object,
                                guint         prop_id,
                                const GValue *value,
                                GParamSpec   *pspec)
{
  FoundryPanelBar *self = FOUNDRY_PANEL_BAR (object);

  switch (prop_id)
    {
    case PROP_SHOW_BOTTOM:
      foundry_panel_bar_set_show_bottom (self, g_value_get_boolean (value));
      break;

    case PROP_SHOW_START:
      foundry_panel_bar_set_show_start (self, g_value_get_boolean (value));
      break;

    case PROP_WORKSPACE:
      foundry_panel_bar_set_workspace (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_panel_bar_class_init (FoundryPanelBarClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->dispose = foundry_panel_bar_dispose;
  object_class->finalize = foundry_panel_bar_finalize;
  object_class->get_property = foundry_panel_bar_get_property;
  object_class->set_property = foundry_panel_bar_set_property;

  properties[PROP_SHOW_START] =
    g_param_spec_boolean ("show-start", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SHOW_BOTTOM] =
    g_param_spec_boolean ("show-bottom", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_WORKSPACE] =
    g_param_spec_object ("workspace", NULL, NULL,
                         FOUNDRY_TYPE_WORKSPACE,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  gtk_widget_class_set_template_from_resource (widget_class, "/app/devsuite/foundry-adw/ui/foundry-panel-bar.ui");
  gtk_widget_class_set_layout_manager_type (widget_class, GTK_TYPE_BOX_LAYOUT);
}

static void
foundry_panel_bar_init (FoundryPanelBar *self)
{
  self->filter = gtk_custom_filter_new (filter_func, self, NULL);
  self->panels = gtk_filter_list_model_new (NULL, g_object_ref (GTK_FILTER (self->filter)));

  g_signal_connect_object (self->panels,
                           "items-changed",
                           G_CALLBACK (foundry_panel_bar_items_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  gtk_widget_init_template (GTK_WIDGET (self));
}

GtkWidget *
foundry_panel_bar_new (void)
{
  return g_object_new (FOUNDRY_TYPE_PANEL_BAR, NULL);
}

gboolean
foundry_panel_bar_get_show_bottom (FoundryPanelBar *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PANEL_BAR (self), FALSE);

  return self->show_bottom;
}

gboolean
foundry_panel_bar_get_show_start (FoundryPanelBar *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PANEL_BAR (self), FALSE);

  return self->show_start;
}

void
foundry_panel_bar_set_show_bottom (FoundryPanelBar *self,
                                   gboolean         show_bottom)
{
  g_return_if_fail (FOUNDRY_IS_PANEL_BAR (self));

  show_bottom = !!show_bottom;

  if (show_bottom != self->show_bottom)
    {
      self->show_bottom = !!show_bottom;
      gtk_filter_changed (GTK_FILTER (self->filter), GTK_FILTER_CHANGE_DIFFERENT);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SHOW_BOTTOM]);
    }
}

void
foundry_panel_bar_set_show_start (FoundryPanelBar *self,
                                  gboolean         show_start)
{
  g_return_if_fail (FOUNDRY_IS_PANEL_BAR (self));

  show_start = !!show_start;

  if (show_start != self->show_start)
    {
      self->show_start = !!show_start;
      gtk_filter_changed (GTK_FILTER (self->filter), GTK_FILTER_CHANGE_DIFFERENT);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SHOW_START]);
    }
}

/**
 * foundry_panel_bar_get_workspace:
 * @self: a [class@FoundryAdw.PanelBar]
 *
 * Returns: (transfer none) (nullable):
 */
FoundryWorkspace *
foundry_panel_bar_get_workspace (FoundryPanelBar *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PANEL_BAR (self), NULL);

  return self->workspace;
}

void
foundry_panel_bar_set_workspace (FoundryPanelBar  *self,
                                 FoundryWorkspace *workspace)
{
  g_return_if_fail (FOUNDRY_IS_PANEL_BAR (self));
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (workspace));

  if (g_set_weak_pointer (&self->workspace, workspace))
    {
      g_autoptr(GListModel) children = NULL;

      if (workspace != NULL)
        children = _foundry_workspace_list_children (workspace);

      gtk_filter_list_model_set_model (self->panels, children);

      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_WORKSPACE]);
    }
}
