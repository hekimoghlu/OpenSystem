/* foundry-workspace.c
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

#include <adwaita.h>
#include <libpanel.h>

#include "foundry-action-responder-group-private.h"
#include "foundry-action-responder-private.h"
#include "foundry-menu-proxy.h"
#include "foundry-multi-reaction-private.h"
#include "foundry-property-reaction-private.h"
#include "foundry-signal-responder-private.h"
#include "foundry-workspace-addin-private.h"
#include "foundry-workspace-child-private.h"
#include "foundry-workspace-private.h"

struct _FoundryWorkspace
{
  GtkWidget                    parent_instance;

  GMenuModel                  *primary_menu;
  GListStore                  *children;
  FoundryContext              *context;
  PeasExtensionSet            *addins;
  FoundryPage                 *active_page;

  PanelDock                   *dock;
  PanelDock                   *subdock;
  AdwWindowTitle              *narrow_panels_title;
  AdwMultiLayoutView          *multi_layout;
  AdwBottomSheet              *narrow_bottom_sheet;
  GtkStack                    *narrow_panels;
  GtkStack                    *narrow_stack;
  AdwTabView                  *narrow_view;
  PanelGrid                   *grid;
  FoundryFrame                *start_frame;
  PanelFrame                  *bottom_frame;
  FoundryActionResponderGroup *narrow_actions;
  AdwBin                      *status_bin;
  AdwBin                      *auxillary_bin;
  AdwBin                      *narrow_auxillary_bin;
  AdwBin                      *wide_auxillary_bin;
  AdwBin                      *titlebar_bin;
  AdwBin                      *sidebar_titlebar_bin;
  AdwBin                      *narrow_titlebar_bin;
};

enum {
  PROP_0,
  PROP_ACTIVE_PAGE,
  PROP_COLLAPSED,
  PROP_CONTEXT,
  PROP_COLLAPSED_TITLEBAR,
  PROP_PRIMARY_MENU,
  PROP_SHOW_AUXILLARY,
  PROP_SHOW_SIDEBAR,
  PROP_SHOW_UTILITIES,
  PROP_SIDEBAR_TITLEBAR,
  PROP_STATUS_WIDGET,
  PROP_TITLEBAR,
  N_PROPS
};

static void buildable_iface_init (GtkBuildableIface *iface);

G_DEFINE_FINAL_TYPE_WITH_CODE (FoundryWorkspace, foundry_workspace, GTK_TYPE_WIDGET,
                               G_IMPLEMENT_INTERFACE (GTK_TYPE_BUILDABLE, buildable_iface_init))

static GParamSpec        *properties[N_PROPS];
static GtkBuildableIface *parent_buildable;

static gboolean
foundry_workspace_is_narrow (FoundryWorkspace *self)
{
  return g_strcmp0 ("narrow", adw_multi_layout_view_get_layout_name (self->multi_layout)) == 0;
}

static void
foundry_workspace_action_narrow_show_menu (GtkWidget  *widget,
                                           const char *action_name,
                                           GVariant   *param)
{
  FoundryWorkspace *self = FOUNDRY_WORKSPACE (widget);

  gtk_stack_set_visible_child_name (self->narrow_stack, "menu");
  adw_bottom_sheet_set_open (self->narrow_bottom_sheet, TRUE);
}

static void
foundry_workspace_layout_changed (FoundryWorkspace *self)
{
  FoundryWorkspaceLayout layout;
  guint n_items;

  g_assert (FOUNDRY_IS_WORKSPACE (self));

  if (foundry_workspace_is_narrow (self))
    layout = FOUNDRY_WORKSPACE_LAYOUT_NARROW;
  else
    layout = FOUNDRY_WORKSPACE_LAYOUT_WIDE;

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->children));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryWorkspaceChild) child = g_list_model_get_item (G_LIST_MODEL (self->children), i);

      foundry_workspace_child_set_layout (child, layout);
    }

  adw_bin_set_child (self->narrow_auxillary_bin, NULL);
  adw_bin_set_child (self->wide_auxillary_bin, NULL);

  if (layout == FOUNDRY_WORKSPACE_LAYOUT_NARROW)
    adw_bin_set_child (self->narrow_auxillary_bin,
                       GTK_WIDGET (self->auxillary_bin));
  else
    adw_bin_set_child (self->wide_auxillary_bin,
                       GTK_WIDGET (self->auxillary_bin));
}

static void
foundry_workspace_notify_narrow_panel (FoundryWorkspace *self)
{
  GtkStackPage *page;
  GtkWidget *visible;

  g_assert (FOUNDRY_IS_WORKSPACE (self));

  if ((visible = gtk_stack_get_visible_child (self->narrow_panels)) &&
      (page = gtk_stack_get_page (self->narrow_panels, visible)))
    g_object_bind_property (page, "title", self->narrow_panels_title, "title", G_BINDING_SYNC_CREATE);
}

static PanelFrame *
foundry_workspace_create_frame_cb (FoundryWorkspace *self,
                                   PanelGrid        *grid)
{
  g_assert (FOUNDRY_IS_WORKSPACE (self));
  g_assert (PANEL_IS_GRID (grid));

  return PANEL_FRAME (foundry_frame_new (FOUNDRY_WORKSPACE_CHILD_PAGE));
}

static gboolean
foundry_workspace_narrow_view_close_page_cb (FoundryWorkspace *self,
                                             AdwTabPage       *tab_page,
                                             AdwTabView       *tab_view)
{
  GtkWidget *child;
  guint n_items;

  g_assert (FOUNDRY_IS_WORKSPACE (self));
  g_assert (ADW_IS_TAB_PAGE (tab_page));
  g_assert (ADW_IS_TAB_VIEW (tab_view));

  child = adw_tab_page_get_child (tab_page);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->children));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryWorkspaceChild) item = g_list_model_get_item (G_LIST_MODEL (self->children), i);
      GtkWidget *frame;
      GtkWidget *wrapper;

      if (child != foundry_workspace_child_get_narrow_widget (item))
        continue;

      /* TODO: Check for confirmation if needed */

      g_debug ("Removing workspace child at index %u", i);
      g_list_store_remove (self->children, i);

      wrapper = foundry_workspace_child_get_wide_widget (item);
      if ((frame = gtk_widget_get_ancestor (wrapper, PANEL_TYPE_FRAME)))
        panel_frame_remove (PANEL_FRAME (frame), PANEL_WIDGET (wrapper));

      adw_tab_view_close_page_finish (tab_view, tab_page, TRUE);

      return GDK_EVENT_STOP;
    }

  return GDK_EVENT_PROPAGATE;
}

static void
foundry_workspace_notify_reveal_start_cb (FoundryWorkspace *self,
                                          GParamSpec       *pspec,
                                          PanelDock        *dock)
{
  g_assert (FOUNDRY_IS_WORKSPACE (self));
  g_assert (PANEL_IS_DOCK (dock));

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SHOW_SIDEBAR]);
}

static void
foundry_workspace_notify_reveal_end_cb (FoundryWorkspace *self,
                                        GParamSpec       *pspec,
                                        PanelDock        *subdock)
{
  g_assert (FOUNDRY_IS_WORKSPACE (self));
  g_assert (PANEL_IS_DOCK (subdock));

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SHOW_AUXILLARY]);
}

static void
foundry_workspace_notify_reveal_bottom_cb (FoundryWorkspace *self,
                                           GParamSpec       *pspec,
                                           PanelDock        *dock)
{
  g_assert (FOUNDRY_IS_WORKSPACE (self));
  g_assert (PANEL_IS_DOCK (dock));

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SHOW_UTILITIES]);
}

static void
foundry_workspace_dispose (GObject *object)
{
  FoundryWorkspace *self = (FoundryWorkspace *)object;
  GtkWidget *child;

  g_clear_object (&self->addins);

  g_list_store_remove_all (self->children);

  gtk_widget_dispose_template (GTK_WIDGET (self), FOUNDRY_TYPE_WORKSPACE);

  while ((child = gtk_widget_get_first_child (GTK_WIDGET (self))))
    gtk_widget_unparent (child);

  g_clear_object (&self->primary_menu);
  g_clear_object (&self->context);

  G_OBJECT_CLASS (foundry_workspace_parent_class)->dispose (object);
}

static void
foundry_workspace_finalize (GObject *object)
{
  FoundryWorkspace *self = (FoundryWorkspace *)object;

  g_clear_object (&self->children);

  G_OBJECT_CLASS (foundry_workspace_parent_class)->finalize (object);
}

static void
foundry_workspace_get_property (GObject    *object,
                                guint       prop_id,
                                GValue     *value,
                                GParamSpec *pspec)
{
  FoundryWorkspace *self = FOUNDRY_WORKSPACE (object);

  switch (prop_id)
    {
    case PROP_ACTIVE_PAGE:
      g_value_set_object (value, foundry_workspace_get_active_page (self));
      break;

    case PROP_COLLAPSED:
      g_value_set_boolean (value, foundry_workspace_get_collapsed (self));
      break;

    case PROP_CONTEXT:
      g_value_set_object (value, foundry_workspace_get_context (self));
      break;

    case PROP_PRIMARY_MENU:
      g_value_set_object (value, foundry_workspace_get_primary_menu (self));
      break;

    case PROP_STATUS_WIDGET:
      g_value_set_object (value, foundry_workspace_get_status_widget (self));
      break;

    case PROP_TITLEBAR:
      g_value_set_object (value, foundry_workspace_get_titlebar (self));
      break;

    case PROP_SHOW_AUXILLARY:
      g_value_set_boolean (value, foundry_workspace_get_show_auxillary (self));
      break;

    case PROP_SHOW_SIDEBAR:
      g_value_set_boolean (value, foundry_workspace_get_show_sidebar (self));
      break;

    case PROP_SHOW_UTILITIES:
      g_value_set_boolean (value, foundry_workspace_get_show_utilities (self));
      break;

    case PROP_SIDEBAR_TITLEBAR:
      g_value_set_object (value, foundry_workspace_get_sidebar_titlebar (self));
      break;

    case PROP_COLLAPSED_TITLEBAR:
      g_value_set_object (value, foundry_workspace_get_collapsed_titlebar (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_workspace_set_property (GObject      *object,
                                guint         prop_id,
                                const GValue *value,
                                GParamSpec   *pspec)
{
  FoundryWorkspace *self = FOUNDRY_WORKSPACE (object);

  switch (prop_id)
    {
    case PROP_CONTEXT:
      foundry_workspace_set_context (self, g_value_get_object (value));
      break;

    case PROP_PRIMARY_MENU:
      foundry_workspace_set_primary_menu (self, g_value_get_object (value));
      break;

    case PROP_SHOW_AUXILLARY:
      foundry_workspace_set_show_auxillary (self, g_value_get_boolean (value));
      break;

    case PROP_SHOW_SIDEBAR:
      foundry_workspace_set_show_sidebar (self, g_value_get_boolean (value));
      break;

    case PROP_SHOW_UTILITIES:
      foundry_workspace_set_show_utilities (self, g_value_get_boolean (value));
      break;

    case PROP_STATUS_WIDGET:
      foundry_workspace_set_status_widget (self, g_value_get_object (value));
      break;

    case PROP_TITLEBAR:
      foundry_workspace_set_titlebar (self, g_value_get_object (value));
      break;

    case PROP_SIDEBAR_TITLEBAR:
      foundry_workspace_set_sidebar_titlebar (self, g_value_get_object (value));
      break;

    case PROP_COLLAPSED_TITLEBAR:
      foundry_workspace_set_collapsed_titlebar (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_workspace_class_init (FoundryWorkspaceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->dispose = foundry_workspace_dispose;
  object_class->finalize = foundry_workspace_finalize;
  object_class->get_property = foundry_workspace_get_property;
  object_class->set_property = foundry_workspace_set_property;

  properties[PROP_ACTIVE_PAGE] =
    g_param_spec_object ("active-page", NULL, NULL,
                         FOUNDRY_TYPE_PAGE,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_COLLAPSED] =
    g_param_spec_boolean ("collapsed", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_CONTEXT] =
    g_param_spec_object ("context", NULL, NULL,
                         FOUNDRY_TYPE_CONTEXT,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PRIMARY_MENU] =
    g_param_spec_object ("primary-menu", NULL, NULL,
                         G_TYPE_MENU_MODEL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_STATUS_WIDGET] =
    g_param_spec_object ("status-widget", NULL, NULL,
                         GTK_TYPE_WIDGET,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SHOW_AUXILLARY] =
    g_param_spec_boolean ("show-auxillary", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SHOW_SIDEBAR] =
    g_param_spec_boolean ("show-sidebar", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SHOW_UTILITIES] =
    g_param_spec_boolean ("show-utilities", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLEBAR] =
    g_param_spec_object ("titlebar", NULL, NULL,
                         GTK_TYPE_WIDGET,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SIDEBAR_TITLEBAR] =
    g_param_spec_object ("sidebar-titlebar", NULL, NULL,
                         GTK_TYPE_WIDGET,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_COLLAPSED_TITLEBAR] =
    g_param_spec_object ("collapsed-titlebar", NULL, NULL,
                         GTK_TYPE_WIDGET,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  gtk_widget_class_set_template_from_resource (widget_class, "/app/devsuite/foundry-adw/ui/foundry-workspace.ui");
  gtk_widget_class_set_layout_manager_type (widget_class, GTK_TYPE_BIN_LAYOUT);

  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, auxillary_bin);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, bottom_frame);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, dock);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, grid);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, multi_layout);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, narrow_actions);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, narrow_auxillary_bin);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, narrow_bottom_sheet);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, narrow_panels);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, narrow_panels_title);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, narrow_stack);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, narrow_titlebar_bin);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, narrow_view);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, sidebar_titlebar_bin);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, start_frame);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, status_bin);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, subdock);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, titlebar_bin);
  gtk_widget_class_bind_template_child (widget_class, FoundryWorkspace, wide_auxillary_bin);

  gtk_widget_class_bind_template_callback (widget_class, foundry_workspace_layout_changed);
  gtk_widget_class_bind_template_callback (widget_class, foundry_workspace_notify_narrow_panel);
  gtk_widget_class_bind_template_callback (widget_class, foundry_workspace_create_frame_cb);
  gtk_widget_class_bind_template_callback (widget_class, foundry_workspace_narrow_view_close_page_cb);

  gtk_widget_class_install_action (widget_class, "workspace.narrow.show-menu", NULL, foundry_workspace_action_narrow_show_menu);

  g_type_ensure (FOUNDRY_TYPE_FRAME);
  g_type_ensure (FOUNDRY_TYPE_MENU_PROXY);
  g_type_ensure (FOUNDRY_TYPE_ACTION_RESPONDER);
  g_type_ensure (FOUNDRY_TYPE_ACTION_RESPONDER_GROUP);
  g_type_ensure (FOUNDRY_TYPE_MULTI_REACTION);
  g_type_ensure (FOUNDRY_TYPE_PROPERTY_REACTION);
  g_type_ensure (FOUNDRY_TYPE_SIGNAL_RESPONDER);
}

static void
foundry_workspace_init (FoundryWorkspace *self)
{
  self->children = g_list_store_new (FOUNDRY_TYPE_WORKSPACE_CHILD);

  gtk_widget_init_template (GTK_WIDGET (self));

  g_signal_connect_object (self->dock,
                           "notify::reveal-start",
                           G_CALLBACK (foundry_workspace_notify_reveal_start_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (self->subdock,
                           "notify::reveal-end",
                           G_CALLBACK (foundry_workspace_notify_reveal_end_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (self->dock,
                           "notify::reveal-bottom",
                           G_CALLBACK (foundry_workspace_notify_reveal_bottom_cb),
                           self,
                           G_CONNECT_SWAPPED);

  gtk_widget_insert_action_group (GTK_WIDGET (self),
                                  "collapsed",
                                  G_ACTION_GROUP (self->narrow_actions));
}

GtkWidget *
foundry_workspace_new (void)
{
  return g_object_new (FOUNDRY_TYPE_WORKSPACE, NULL);
}

/**
 * foundry_workspace_get_primary_menu:
 * @self: a [class@FoundryAdw.Workspace]
 *
 * Returns: (transfer none) (nullable):
 */
GMenuModel *
foundry_workspace_get_primary_menu (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), NULL);

  return self->primary_menu;
}

void
foundry_workspace_set_primary_menu (FoundryWorkspace *self,
                                    GMenuModel       *primary_menu)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (!primary_menu || G_IS_MENU_MODEL (primary_menu));

  if (g_set_object (&self->primary_menu, primary_menu))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_PRIMARY_MENU]);
}

static FoundryWorkspaceChild *
foundry_workspace_find_child (FoundryWorkspace *self,
                              GtkWidget        *widget)
{
  guint n_items;

  g_assert (FOUNDRY_IS_WORKSPACE (self));
  g_assert (GTK_IS_WIDGET (widget));

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->children));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryWorkspaceChild) child = g_list_model_get_item (G_LIST_MODEL (self->children), i);

      if (widget == foundry_workspace_child_get_child (child))
        return g_steal_pointer (&child);
    }

  return NULL;
}

static void
foundry_workspace_raise_page_cb (FoundryWorkspace *self,
                                 FoundryPage      *page)
{
  g_autoptr(FoundryWorkspaceChild) child = NULL;
  g_autofree char *title = NULL;

  g_assert (FOUNDRY_IS_WORKSPACE (self));
  g_assert (FOUNDRY_IS_PAGE (page));

  title = foundry_page_dup_title (page);
  g_debug ("Raising page `%s` (%s)", title, G_OBJECT_TYPE_NAME (page));

  if (!(child = foundry_workspace_find_child (self, GTK_WIDGET (page))))
    {
      g_debug ("Failed to find child to raise");
      return;
    }

  if (foundry_workspace_is_narrow (self))
    {
      GtkWidget *wrapper = foundry_workspace_child_get_narrow_widget (child);
      AdwTabPage *tab_page = adw_tab_view_get_page (self->narrow_view, wrapper);

      if (tab_page != NULL)
        {
          adw_tab_view_set_selected_page (self->narrow_view, tab_page);
          adw_bottom_sheet_set_open (self->narrow_bottom_sheet, FALSE);
        }
    }
  else
    {
      GtkWidget *wrapper = foundry_workspace_child_get_wide_widget (child);

      g_assert (PANEL_IS_WIDGET (wrapper));

      panel_widget_raise (PANEL_WIDGET (wrapper));
    }
}

static gboolean
icon_to_icon_name (GBinding     *binding,
                   const GValue *from,
                   GValue       *to,
                   gpointer      user_data)
{
  GIcon *icon;
  const char * const *names;

  if ((icon = g_value_get_object (from)) &&
      G_IS_THEMED_ICON (icon) &&
      (names = g_themed_icon_get_names (G_THEMED_ICON (icon))))
    g_value_set_string (to, names[0]);

  return TRUE;
}

static void
foundry_workspace_add_panel (FoundryWorkspace *self,
                             FoundryPanel     *panel,
                             gboolean          sidebar)
{
  g_autoptr(FoundryWorkspaceChild) child = NULL;
  GtkStackPage *page;
  GtkWidget *wrapper;

  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (FOUNDRY_IS_PANEL (panel));

  child = foundry_workspace_child_new (FOUNDRY_WORKSPACE_CHILD_PANEL, sidebar ? PANEL_AREA_START : PANEL_AREA_BOTTOM);
  foundry_workspace_child_set_child (child, GTK_WIDGET (panel));
  g_object_bind_property (panel, "title", child, "title", G_BINDING_SYNC_CREATE);
  g_object_bind_property (panel, "icon", child, "icon", G_BINDING_SYNC_CREATE);
  g_list_store_append (self->children, child);

  wrapper = foundry_workspace_child_get_wide_widget (child);

  if (sidebar)
    panel_frame_add (PANEL_FRAME (self->start_frame), PANEL_WIDGET (wrapper));
  else
    panel_frame_add (self->bottom_frame, PANEL_WIDGET (wrapper));

  wrapper = foundry_workspace_child_get_narrow_widget (child);
  page = gtk_stack_add_child (self->narrow_panels, wrapper);
  g_object_bind_property (panel, "title", page, "title", G_BINDING_SYNC_CREATE);
  g_object_bind_property_full (panel, "icon", page, "icon-name",
                               G_BINDING_SYNC_CREATE,
                               icon_to_icon_name, NULL, NULL, NULL);
}

void
foundry_workspace_remove_panel (FoundryWorkspace *self,
                                FoundryPanel     *panel)
{
  GListModel *model;
  GtkWidget *wrapper;
  guint n_items;

  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (FOUNDRY_IS_PANEL (panel));

  model = G_LIST_MODEL (self->children);
  n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryWorkspaceChild) item = g_list_model_get_item (model, i);

      if (GTK_WIDGET (panel) == foundry_workspace_child_get_child (item))
        {
          GtkWidget *frame;

          g_list_store_remove (self->children, i);

          wrapper = foundry_workspace_child_get_narrow_widget (item);
          gtk_stack_remove (self->narrow_panels, wrapper);

          wrapper = foundry_workspace_child_get_wide_widget (item);
          if ((frame = gtk_widget_get_ancestor (wrapper, PANEL_TYPE_FRAME)))
            panel_frame_remove (PANEL_FRAME (frame), PANEL_WIDGET (wrapper));

          break;
        }
    }
}

void
foundry_workspace_add_sidebar_panel (FoundryWorkspace *self,
                                     FoundryPanel     *panel)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (FOUNDRY_IS_PANEL (panel));

  foundry_workspace_add_panel (self, panel, TRUE);
}

void
foundry_workspace_add_bottom_panel (FoundryWorkspace *self,
                                    FoundryPanel     *panel)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (FOUNDRY_IS_PANEL (panel));

  foundry_workspace_add_panel (self, panel, FALSE);
}

void
foundry_workspace_add_page (FoundryWorkspace *self,
                            FoundryPage      *page)
{
  g_autoptr(FoundryWorkspaceChild) child = NULL;
  AdwTabPage *tab_page;
  GtkWidget *wrapper;

  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (FOUNDRY_IS_PAGE (page));

  child = foundry_workspace_child_new (FOUNDRY_WORKSPACE_CHILD_PAGE, PANEL_AREA_CENTER);
  foundry_workspace_child_set_child (child, GTK_WIDGET (page));
  g_object_bind_property (page, "title", child, "title", G_BINDING_SYNC_CREATE);
  g_object_bind_property (page, "subtitle", child, "subtitle", G_BINDING_SYNC_CREATE);
  g_object_bind_property (page, "icon", child, "icon", G_BINDING_SYNC_CREATE);
  g_object_bind_property (page, "can-save", child, "modified", G_BINDING_SYNC_CREATE);
  g_list_store_append (self->children, child);

  wrapper = foundry_workspace_child_get_wide_widget (child);
  panel_grid_add (self->grid, PANEL_WIDGET (wrapper));

  wrapper = foundry_workspace_child_get_narrow_widget (child);
  tab_page = adw_tab_view_add_page (self->narrow_view, wrapper, NULL);
  g_object_bind_property (page, "title", tab_page, "title", G_BINDING_SYNC_CREATE);

  if (foundry_workspace_is_narrow (self))
    foundry_workspace_child_set_layout (child, FOUNDRY_WORKSPACE_LAYOUT_NARROW);

  g_signal_connect_object (page,
                           "raise",
                           G_CALLBACK (foundry_workspace_raise_page_cb),
                           self,
                           G_CONNECT_SWAPPED);

  if (self->active_page == NULL)
    _foundry_workspace_set_active_page (self, page);
}

void
foundry_workspace_remove_page (FoundryWorkspace *self,
                               FoundryPage      *page)
{
  GListModel *model;
  GtkWidget *wrapper;
  guint n_items;

  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (FOUNDRY_IS_PAGE (page));

  g_signal_handlers_disconnect_by_func (page,
                                        G_CALLBACK (foundry_workspace_raise_page_cb),
                                        self);

  model = G_LIST_MODEL (self->children);
  n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryWorkspaceChild) item = g_list_model_get_item (model, i);

      if (GTK_WIDGET (page) == foundry_workspace_child_get_child (item))
        {
          AdwTabPage *tab_page;
          GtkWidget *frame;

          g_debug ("Removing workspace child at index %u", i);
          g_list_store_remove (self->children, i);

          wrapper = foundry_workspace_child_get_narrow_widget (item);
          tab_page = adw_tab_view_get_page (self->narrow_view, wrapper);
          adw_tab_view_close_page (self->narrow_view, tab_page);

          wrapper = foundry_workspace_child_get_wide_widget (item);
          if ((frame = gtk_widget_get_ancestor (wrapper, PANEL_TYPE_FRAME)))
            panel_frame_remove (PANEL_FRAME (frame), PANEL_WIDGET (wrapper));

          break;
        }
    }
}

static void
foundry_workspace_add_child (GtkBuildable *buildable,
                             GtkBuilder   *builder,
                             GObject      *object,
                             const char   *type)
{
  FoundryWorkspace *self = (FoundryWorkspace *)buildable;

  g_assert (FOUNDRY_IS_WORKSPACE (self));
  g_assert (GTK_IS_BUILDER (builder));

  if (FOUNDRY_IS_PANEL (object))
    {
      if (g_strcmp0 (type, "bottom") == 0)
        foundry_workspace_add_bottom_panel (self, FOUNDRY_PANEL (object));
      else
        foundry_workspace_add_sidebar_panel (self, FOUNDRY_PANEL (object));
    }
  else if (FOUNDRY_IS_PAGE (object))
    foundry_workspace_add_page (self, FOUNDRY_PAGE (object));
  else if (g_strcmp0 (type, "status") == 0 && GTK_IS_WIDGET (object))
    foundry_workspace_set_status_widget (self, GTK_WIDGET (object));
  else
    parent_buildable->add_child (buildable, builder, object, type);
}

static void
buildable_iface_init (GtkBuildableIface *iface)
{
  iface->add_child = foundry_workspace_add_child;
  parent_buildable = g_type_interface_peek_parent (iface);
}

void
_foundry_workspace_frame_page_closed (FoundryWorkspace          *self,
                                      FoundryFrame              *frame,
                                      PanelWidget               *child,
                                      FoundryWorkspaceChildKind  kind)
{
  guint n_items;

  g_assert (FOUNDRY_IS_WORKSPACE (self));
  g_assert (FOUNDRY_IS_FRAME (frame));
  g_assert (PANEL_IS_WIDGET (child));

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->children));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryWorkspaceChild) item = g_list_model_get_item (G_LIST_MODEL (self->children), i);
      PanelWidget *wide = PANEL_WIDGET (foundry_workspace_child_get_wide_widget (item));
      GtkWidget *wrapper;

      if (wide != child)
        continue;

      g_debug ("Removing workspace child at index %u", i);
      g_list_store_remove (self->children, i);

      wrapper = foundry_workspace_child_get_narrow_widget (item);

      if (kind == FOUNDRY_WORKSPACE_CHILD_PAGE)
        {
          AdwTabPage *tab_page;

          tab_page = adw_tab_view_get_page (self->narrow_view, wrapper);
          adw_tab_view_close_page (self->narrow_view, tab_page);
        }
      else
        {
          gtk_stack_remove (self->narrow_panels, wrapper);
        }

      break;
    }
}

/**
 * foundry_workspace_get_context:
 * @self: a [class@FoundryAdw.Workspace]
 *
 * Returns: (transfer none) (nullable):
 */
FoundryContext *
foundry_workspace_get_context (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), NULL);

  return self->context;
}

static void
foundry_workspace_addin_added (PeasExtensionSet *set,
                               PeasPluginInfo   *plugin_info,
                               GObject          *addin,
                               gpointer          user_data)
{
  FoundryWorkspace *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_WORKSPACE_ADDIN (addin));
  g_assert (FOUNDRY_IS_WORKSPACE (self));

  g_debug ("Adding FoundryWorkspaceAddin of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (_foundry_workspace_addin_load (FOUNDRY_WORKSPACE_ADDIN (addin), self));
}

static void
foundry_workspace_addin_removed (PeasExtensionSet *set,
                                 PeasPluginInfo   *plugin_info,
                                 GObject          *addin,
                                 gpointer          user_data)
{
  FoundryWorkspace *self = user_data;

  g_assert (PEAS_IS_EXTENSION_SET (set));
  g_assert (PEAS_IS_PLUGIN_INFO (plugin_info));
  g_assert (FOUNDRY_IS_WORKSPACE_ADDIN (addin));
  g_assert (FOUNDRY_IS_WORKSPACE (self));

  g_debug ("Removing FoundryWorkspaceAddin of type %s", G_OBJECT_TYPE_NAME (addin));

  dex_future_disown (_foundry_workspace_addin_unload (FOUNDRY_WORKSPACE_ADDIN (addin)));
}

void
foundry_workspace_set_context (FoundryWorkspace *self,
                               FoundryContext   *context)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (!context || FOUNDRY_IS_CONTEXT (context));

  if (g_set_object (&self->context, context))
    {
      g_clear_object (&self->addins);

      if (context != NULL)
        {
          self->addins = peas_extension_set_new (peas_engine_get_default (),
                                                 FOUNDRY_TYPE_WORKSPACE_ADDIN,
                                                 NULL, NULL);

          g_signal_connect_object (self->addins,
                                   "extension-added",
                                   G_CALLBACK (foundry_workspace_addin_added),
                                   self,
                                   0);
          g_signal_connect_object (self->addins,
                                   "extension-removed",
                                   G_CALLBACK (foundry_workspace_addin_removed),
                                   self,
                                   0);

          peas_extension_set_foreach (self->addins,
                                      foundry_workspace_addin_added,
                                      self);
        }

      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CONTEXT]);
    }
}

/**
 * foundry_workspace_get_status_widget:
 * @self: a [class@FoundryAdw.Workspace]
 *
 * Returns: (transfer none) (nullable):
 */
GtkWidget *
foundry_workspace_get_status_widget (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), NULL);

  return adw_bin_get_child (self->status_bin);
}

void
foundry_workspace_set_status_widget (FoundryWorkspace *self,
                                     GtkWidget        *status_widget)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));

  if (status_widget == foundry_workspace_get_status_widget (self))
    return;

  adw_bin_set_child (self->status_bin, status_widget);
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_STATUS_WIDGET]);
}


/**
 * foundry_workspace_get_active_page:
 * @self: a [class@FoundryAdw.Workspace]
 *
 * Returns: (transfer none) (nullable):
 */
FoundryPage *
foundry_workspace_get_active_page (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), NULL);

  return self->active_page;
}

void
_foundry_workspace_set_active_page (FoundryWorkspace *self,
                                    FoundryPage      *page)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (!page || FOUNDRY_IS_PAGE (page));

  if (g_set_object (&self->active_page, page))
    {
      adw_bin_set_child (self->auxillary_bin, NULL);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ACTIVE_PAGE]);
    }
}

/**
 * foundry_workspace_foreach_page:
 * @self: a [class@FoundryAdw.Workspace]
 * @callback: (scope call):
 *
 * Calls @callback for every [class@FoundryAdw.Page] in the workspace.
 */
void
foundry_workspace_foreach_page (FoundryWorkspace *self,
                                GFunc             callback,
                                gpointer          user_data)
{
  guint n_items;

  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (callback != NULL);

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->children));

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryWorkspaceChild) child = g_list_model_get_item (G_LIST_MODEL (self->children), i);
      GtkWidget *widget = foundry_workspace_child_get_child (child);

      g_assert (!widget ||
                FOUNDRY_IS_PAGE (widget) ||
                FOUNDRY_IS_PANEL (widget));

      if (FOUNDRY_IS_PAGE (widget))
        callback (widget, user_data);
    }
}

/**
 * foundry_workspace_get_titlebar:
 *
 * Returns: (transfer none) (nullable):
 */
GtkWidget *
foundry_workspace_get_titlebar (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), NULL);

  return adw_bin_get_child (self->titlebar_bin);
}

void
foundry_workspace_set_titlebar (FoundryWorkspace *self,
                                GtkWidget        *titlebar)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (!titlebar || GTK_IS_WIDGET (titlebar));
  g_return_if_fail (!titlebar || gtk_widget_get_parent (titlebar) == NULL);

  if (foundry_workspace_get_titlebar (self) == titlebar)
    return;

  adw_bin_set_child (self->titlebar_bin, titlebar);
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_TITLEBAR]);
}

/**
 * foundry_workspace_get_collapsed_titlebar:
 *
 * Returns: (transfer none) (nullable):
 */
GtkWidget *
foundry_workspace_get_collapsed_titlebar (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), NULL);

  return adw_bin_get_child (self->narrow_titlebar_bin);
}

void
foundry_workspace_set_collapsed_titlebar (FoundryWorkspace *self,
                                          GtkWidget        *collapsed_titlebar)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (!collapsed_titlebar || GTK_IS_WIDGET (collapsed_titlebar));
  g_return_if_fail (!collapsed_titlebar || gtk_widget_get_parent (collapsed_titlebar) == NULL);

  if (foundry_workspace_get_collapsed_titlebar (self) == collapsed_titlebar)
    return;

  adw_bin_set_child (self->narrow_titlebar_bin, collapsed_titlebar);
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_COLLAPSED_TITLEBAR]);
}

/**
 * foundry_workspace_get_sidebar_titlebar:
 *
 * Returns: (transfer none) (nullable):
 */
GtkWidget *
foundry_workspace_get_sidebar_titlebar (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), NULL);

  return adw_bin_get_child (self->sidebar_titlebar_bin);
}

void
foundry_workspace_set_sidebar_titlebar (FoundryWorkspace *self,
                                        GtkWidget        *sidebar_titlebar)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));
  g_return_if_fail (!sidebar_titlebar || GTK_IS_WIDGET (sidebar_titlebar));
  g_return_if_fail (!sidebar_titlebar || gtk_widget_get_parent (sidebar_titlebar) == NULL);

  if (foundry_workspace_get_sidebar_titlebar (self) == sidebar_titlebar)
    return;

  adw_bin_set_child (self->sidebar_titlebar_bin, sidebar_titlebar);
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SIDEBAR_TITLEBAR]);
}

/**
 * foundry_workspace_get_collapsed:
 *
 * Gets if the workspace is in collapsed form, meaning a narrow
 * representation of the window. In collapsed form, the sidebar
 * and main contents are not visible but instead a condensed form
 * of the content with access to panels is shown.
 */
gboolean
foundry_workspace_get_collapsed (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), FALSE);

  return foundry_workspace_is_narrow (self);
}

/**
 * foundry_workspace_get_show_sidebar:
 *
 * Gets if the sidebar should be shown when the workspace is not collapsed.
 */
gboolean
foundry_workspace_get_show_sidebar (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), FALSE);

  return panel_dock_get_reveal_start (self->dock);
}

/**
 * foundry_workspace_set_show_sidebar:
 *
 * Sets if the sidebar should be shown when the workspace is not collapsed.
 */
void
foundry_workspace_set_show_sidebar (FoundryWorkspace *self,
                                    gboolean          show_sidebar)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));

  show_sidebar = !!show_sidebar;

  if (show_sidebar != foundry_workspace_get_show_sidebar (self))
    panel_dock_set_reveal_start (self->dock, !!show_sidebar);
}

/**
 * foundry_workspace_get_show_auxillary:
 *
 * Gets if the auxillary should be shown when the workspace is not collapsed.
 */
gboolean
foundry_workspace_get_show_auxillary (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), FALSE);

  return panel_dock_get_reveal_end (self->subdock);
}

/**
 * foundry_workspace_set_show_auxillary:
 *
 * Sets if the auxillary should be shown when the workspace is not collapsed.
 */
void
foundry_workspace_set_show_auxillary (FoundryWorkspace *self,
                                      gboolean          show_auxillary)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));

  show_auxillary = !!show_auxillary;

  if (show_auxillary != foundry_workspace_get_show_auxillary (self))
    panel_dock_set_reveal_end (self->subdock, !!show_auxillary);
}

/**
 * foundry_workspace_get_show_utilities:
 *
 * Gets if the utilities should be shown when the workspace is not collapsed.
 */
gboolean
foundry_workspace_get_show_utilities (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), FALSE);

  return panel_dock_get_reveal_bottom (self->dock);
}

/**
 * foundry_workspace_set_show_utilities:
 *
 * Sets if the utilities should be shown when the workspace is not collapsed.
 */
void
foundry_workspace_set_show_utilities (FoundryWorkspace *self,
                                      gboolean          show_utilities)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE (self));

  show_utilities = !!show_utilities;

  if (show_utilities != foundry_workspace_get_show_utilities (self))
    panel_dock_set_reveal_bottom (self->dock, !!show_utilities);
}

GListModel *
_foundry_workspace_list_children (FoundryWorkspace *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE (self), NULL);

  return g_object_ref (G_LIST_MODEL (self->children));
}
