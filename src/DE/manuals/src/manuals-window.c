/* manuals-window.c
 *
 * Copyright 2025 Christian Hergert
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
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "config.h"

#include <glib/gi18n.h>

#include <libpanel.h>

#include "manuals-application.h"
#include "manuals-bundle-dialog.h"
#include "manuals-path-bar.h"
#include "manuals-search-row.h"
#include "manuals-tree-expander.h"
#include "manuals-window.h"
#include "manuals-wrapped-model.h"

#define MAX_SEARCH_RESULTS 1000

struct _ManualsWindow
{
  AdwApplicationWindow    parent_instance;

  GSignalGroup           *visible_tab_signals;

  PanelDock              *dock;
  PanelStatusbar         *statusbar;
  AdwTabView             *tab_view;
  AdwWindowTitle         *title;
  GtkSearchEntry         *search_entry;
  GSettings              *settings;
  AdwToolbarView         *sidebar;
  GtkStack               *stack;
  GtkStack               *sidebar_stack;
  GtkNoSelection         *selection;
  GtkListView            *list_view;
  GtkListView            *search_list_view;
  GtkSingleSelection     *search_selection;
  AdwNavigationSplitView *split_view;

  guint                   stamp;

  guint                   disposed : 1;
};

G_DEFINE_FINAL_TYPE (ManualsWindow, manuals_window, ADW_TYPE_APPLICATION_WINDOW)

enum {
  PROP_0,
  PROP_VISIBLE_TAB,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static gboolean
invert_boolean (gpointer object,
                gboolean value)
{
  return !value;
}

static void
manuals_window_update_actions (ManualsWindow *self)
{
  ManualsTab *tab;
  gboolean can_go_forward = FALSE;
  gboolean can_go_back = FALSE;

  g_assert (MANUALS_IS_WINDOW (self));

  if (self->disposed)
    return;

  if ((tab = manuals_window_get_visible_tab (self)))
    {
      can_go_back = manuals_tab_can_go_back (tab);
      can_go_forward = manuals_tab_can_go_forward (tab);
    }

  gtk_widget_action_set_enabled (GTK_WIDGET (self), "tab.go-back", can_go_back);
  gtk_widget_action_set_enabled (GTK_WIDGET (self), "tab.go-forward", can_go_forward);
}

static void
manuals_window_update_stack_child (ManualsWindow *self)
{
  g_autoptr(GtkSelectionModel) pages = NULL;
  const char *old_child_name;
  const char *child_name;
  gboolean import_active;
  gboolean has_tabs;

  g_assert (MANUALS_IS_WINDOW (self));

  import_active = manuals_application_get_import_active (MANUALS_APPLICATION_DEFAULT);
  pages = adw_tab_view_get_pages (self->tab_view);
  has_tabs = g_list_model_get_n_items (G_LIST_MODEL (pages)) > 0;
  old_child_name = gtk_stack_get_visible_child_name (self->stack);

  if (import_active)
    child_name = "loading";
  else if (has_tabs)
    child_name = "tabs";
  else
    child_name = "empty";

  gtk_stack_set_visible_child_name (self->stack, child_name);

  gtk_widget_set_visible (GTK_WIDGET (self->statusbar),
                          g_str_equal (child_name, "tabs"));

  if (self->sidebar != NULL)
    gtk_widget_set_sensitive (GTK_WIDGET (self->sidebar),
                              !g_str_equal (child_name, "loading"));

  if (g_strcmp0 (old_child_name, child_name) != 0 &&
      g_str_equal (child_name, "empty"))
    {
      panel_dock_set_reveal_start (self->dock, TRUE);
      gtk_widget_grab_focus (GTK_WIDGET (self->search_entry));
    }
}

static void
manuals_window_notify_import_active_cb (ManualsWindow      *self,
                                        GParamSpec         *pspec,
                                        ManualsApplication *app)
{
  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (MANUALS_IS_APPLICATION (app));

  manuals_window_update_stack_child (self);
}

static gboolean
on_tab_view_close_page_cb (ManualsWindow *self,
                           AdwTabPage    *page,
                           AdwTabView    *tab_view)
{
  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (ADW_IS_TAB_PAGE (page));
  g_assert (ADW_IS_TAB_VIEW (tab_view));

  adw_tab_view_close_page_finish (tab_view, page, TRUE);

  manuals_window_update_stack_child (self);

  return GDK_EVENT_STOP;
}

static ManualsTab *
manuals_window_new_tab_for_uri_from_current_tab (ManualsWindow *self,
                                                 const char    *uri)
{
  g_autoptr(FoundryDocumentation) navigatable = NULL;
  ManualsTab *original_tab;
  ManualsTab *new_tab;

  g_assert (MANUALS_IS_WINDOW (self));

  if (g_strcmp0 ("file", g_uri_peek_scheme (uri)) != 0)
    {
      g_autoptr(GtkUriLauncher) launcher = gtk_uri_launcher_new (uri);
      gtk_uri_launcher_launch (launcher, GTK_WINDOW (self), NULL, NULL, NULL);
      return NULL;
    }

  original_tab = manuals_window_get_visible_tab (self);

  if (original_tab)
    new_tab = manuals_tab_duplicate (original_tab);
  else
    new_tab = manuals_tab_new ();

  manuals_tab_load_uri (new_tab, uri);

  return new_tab;
}

static void
manuals_window_open_uri_in_new_tab_action (GSimpleAction *action,
                                           GVariant      *param,
                                           gpointer       user_data)
{
  ManualsWindow *self = user_data;
  ManualsTab *new_tab;

  g_assert (G_IS_SIMPLE_ACTION (action));
  g_assert (param);
  g_assert (MANUALS_IS_WINDOW (self));

  new_tab = manuals_window_new_tab_for_uri_from_current_tab (self, g_variant_get_string (param, NULL));
  if (new_tab)
    manuals_window_add_tab (self, new_tab);
}

static void
manuals_window_open_uri_in_new_window_action (GSimpleAction *action,
                                              GVariant      *param,
                                              gpointer       user_data)
{
  ManualsWindow *self = user_data;
  ManualsWindow *new_window;
  ManualsTab *new_tab;

  g_assert (G_IS_SIMPLE_ACTION (action));
  g_assert (param);
  g_assert (MANUALS_IS_WINDOW (self));

  new_tab = manuals_window_new_tab_for_uri_from_current_tab (self, g_variant_get_string (param, NULL));
  if (new_tab)
    {
      new_window = manuals_window_new ();
      manuals_window_add_tab (new_window, new_tab);
      gtk_window_present (GTK_WINDOW (new_window));
    }
}

static void
manuals_window_tab_view_notify_selected_page_cb (ManualsWindow *self,
                                                 GParamSpec    *pspec,
                                                 AdwTabView    *tab_view)
{
  ManualsTab *tab;

  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (ADW_IS_TAB_VIEW (tab_view));

  adw_window_title_set_title (self->title, "");

  tab = manuals_window_get_visible_tab (self);
  g_signal_group_set_target (self->visible_tab_signals, tab);

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_VISIBLE_TAB]);
}

static void
manuals_window_tab_go_back_action (GtkWidget  *widget,
                                   const char *action_name,
                                   GVariant   *param)
{
  manuals_tab_go_back (manuals_window_get_visible_tab (MANUALS_WINDOW (widget)));
}

static void
manuals_window_tab_go_forward_action (GtkWidget  *widget,
                                      const char *action_name,
                                      GVariant   *param)
{
  manuals_tab_go_forward (manuals_window_get_visible_tab (MANUALS_WINDOW (widget)));
}

static void
manuals_window_tab_focus_search_action (GtkWidget  *widget,
                                        const char *action_name,
                                        GVariant   *param)
{
  ManualsWindow *self = MANUALS_WINDOW (widget);
  ManualsTab *tab;

  g_assert (MANUALS_IS_WINDOW (self));

  if ((tab = manuals_window_get_visible_tab (self)))
    manuals_tab_focus_search (tab);
}

static void
manuals_window_tab_new_action (GtkWidget  *widget,
                               const char *action_name,
                               GVariant   *param)
{
  ManualsWindow *self = MANUALS_WINDOW (widget);
  ManualsTab *tab;

  g_assert (MANUALS_IS_WINDOW (self));

  if (!(tab = manuals_window_get_visible_tab (self)))
    return;

  tab = manuals_tab_duplicate (tab);

  manuals_window_add_tab (self, tab);
  manuals_window_set_visible_tab (self, tab);
}

static void
manuals_window_tab_close_action (GtkWidget  *widget,
                                 const char *action_name,
                                 GVariant   *param)
{
  ManualsWindow *self = MANUALS_WINDOW (widget);
  ManualsTab *tab;
  AdwTabPage *page;

  g_assert (MANUALS_IS_WINDOW (self));

  if (!(tab = manuals_window_get_visible_tab (self)))
    {
      gtk_window_destroy (GTK_WINDOW (self));
      return;
    }

  if ((page = adw_tab_view_get_page (self->tab_view, GTK_WIDGET (tab))))
    adw_tab_view_close_page (self->tab_view, page);
}

static void
manuals_window_sidebar_focus_search_action (GtkWidget  *widget,
                                            const char *action_name,
                                            GVariant   *param)
{
  ManualsWindow *self = MANUALS_WINDOW (widget);

  g_assert (MANUALS_IS_WINDOW (self));

  panel_dock_set_reveal_start (self->dock, TRUE);
  gtk_widget_grab_focus (GTK_WIDGET (self->search_entry));
}

static void
manuals_window_show_bundle_dialog_action (GtkWidget  *widget,
                                          const char *action_name,
                                          GVariant   *param)
{
  ManualsWindow *self = MANUALS_WINDOW (widget);
  ManualsBundleDialog *dialog;

  g_assert (MANUALS_IS_WINDOW (self));

  dialog = manuals_bundle_dialog_new ();
  manuals_bundle_dialog_present (dialog, GTK_WIDGET (self));
}

static void
manuals_window_save_sidebar_width (ManualsWindow *self,
                                   GParamSpec    *pspec,
                                   GtkWidget     *widget)
{
  int width;

  g_assert (MANUALS_IS_WINDOW (self));

  g_object_get (self->dock, "start-width", &width, NULL);
  g_settings_set_uint (self->settings, "sidebar-width", MAX (100, width));
}

static void
manuals_window_size_allocate (GtkWidget *widget,
                              int        width,
                              int        height,
                              int        baseline)
{
  ManualsWindow *self = MANUALS_WINDOW (widget);

  GTK_WIDGET_CLASS (manuals_window_parent_class)->size_allocate (widget, width, height, baseline);

  if (!gtk_window_is_maximized (GTK_WINDOW (self)))
    g_settings_set (self->settings, "window-size", "(ii)", width, height);
}

static void
manuals_window_notify_maximize (ManualsWindow *self)
{
  g_assert (MANUALS_IS_WINDOW (self));

  g_settings_set_boolean (self->settings,
                          "maximized",
                          gtk_window_is_maximized (GTK_WINDOW (self)));
}

static void
manuals_window_constructed (GObject *object)
{
  ManualsWindow *self = (ManualsWindow *)object;
  GtkWidget *widget;
  guint width, height;

  G_OBJECT_CLASS (manuals_window_parent_class)->constructed (object);

#ifdef DEVELOPMENT_BUILD
  gtk_widget_add_css_class (GTK_WIDGET (object), "devel");
#endif

  g_settings_get (self->settings, "window-size", "(ii)", &width, &height);
  gtk_window_set_default_size (GTK_WINDOW (self), width, height);

  panel_dock_set_start_width (self->dock,
                              g_settings_get_uint (self->settings, "sidebar-width"));

  widget = gtk_widget_get_ancestor (GTK_WIDGET (self->search_entry),
                                    g_type_from_name ("PanelResizer"));

  g_signal_connect_object (widget,
                           "notify::drag-position",
                           G_CALLBACK (manuals_window_save_sidebar_width),
                           self,
                           G_CONNECT_SWAPPED);

  if (g_settings_get_boolean (self->settings, "maximized"))
    gtk_window_maximize (GTK_WINDOW (self));

  g_signal_connect (self,
                    "notify::maximized",
                    G_CALLBACK (manuals_window_notify_maximize),
                    NULL);

  gtk_widget_grab_focus (GTK_WIDGET (self->search_entry));

  g_signal_connect_object (MANUALS_APPLICATION_DEFAULT,
                           "notify::import-active",
                           G_CALLBACK (manuals_window_notify_import_active_cb),
                           self,
                           G_CONNECT_SWAPPED);
  manuals_window_notify_import_active_cb (self, NULL, MANUALS_APPLICATION_DEFAULT);
}

static GListModel *
manuals_window_create_child_model (gpointer item,
                                   gpointer user_data)
{
  FoundryDocumentation *doc = item;

  g_assert (FOUNDRY_IS_DOCUMENTATION (doc));

  if (!foundry_documentation_has_children (doc))
    return NULL;

  return manuals_wrapped_model_new (foundry_documentation_find_children (doc));
}

static DexFuture *
manuals_window_reload_fiber (gpointer user_data)
{
  ManualsWindow *self = user_data;
  g_autoptr(FoundryDocumentationManager) documentation_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListModel) children = NULL;

  g_assert (MANUALS_IS_WINDOW (self));

  if ((context = dex_await_object (manuals_application_load_foundry (MANUALS_APPLICATION_DEFAULT), NULL)) &&
      (documentation_manager = foundry_context_dup_documentation_manager (context)) &&
      dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (documentation_manager)), NULL) &&
      (children = dex_await_object (foundry_documentation_manager_list_children (documentation_manager, NULL), NULL)))
    {
      g_autoptr(GtkTreeListModel) tree = NULL;

      g_debug ("Window has new list of documentation, reloading tree");

      tree = gtk_tree_list_model_new (g_object_ref (children),
                                      FALSE,
                                      FALSE,
                                      manuals_window_create_child_model,
                                      NULL, NULL);
      gtk_no_selection_set_model (self->selection, G_LIST_MODEL (tree));

      return NULL;
    }

  g_debug ("Failed to query updated documentation");

  return NULL;
}

static void
manuals_window_reload (ManualsWindow *self)
{
  g_assert (MANUALS_IS_WINDOW (self));

  dex_future_disown (dex_scheduler_spawn (NULL, 0,
                                          manuals_window_reload_fiber,
                                          g_object_ref (self),
                                          g_object_unref));
}

static void
manuals_window_list_view_activate_cb (ManualsWindow *self,
                                      guint          position,
                                      GtkListView   *list_view)
{
  g_autoptr(FoundryDocumentation) documentation = NULL;
  g_autoptr(GtkTreeListRow) row = NULL;
  GtkSelectionModel *model;

  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (GTK_IS_LIST_VIEW (list_view));

  model = gtk_list_view_get_model (list_view);
  row = g_list_model_get_item (G_LIST_MODEL (model), position);
  documentation = gtk_tree_list_row_get_item (row);

  manuals_window_navigate_to (self, documentation, FALSE);
}

static DexFuture *
manuals_window_search_fiber (ManualsWindow *self,
                             const char    *text,
                             guint          stamp)
{
  g_autoptr(FoundryDocumentationManager) manager = NULL;
  g_autoptr(FoundryDocumentationMatches) matches = NULL;
  g_autoptr(FoundryDocumentationQuery) query = NULL;
  g_autoptr(FoundryContext) context = NULL;

  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (text != NULL);

  query = foundry_documentation_query_new ();
  foundry_documentation_query_set_keyword (query, text);

  if ((context = dex_await_object (manuals_application_load_foundry (MANUALS_APPLICATION_DEFAULT), NULL)) &&
      (manager = foundry_context_dup_documentation_manager (context)) &&
      dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (manager)), NULL) &&
      (matches = dex_await_object (foundry_documentation_manager_query (manager, query), NULL)))
    {
      g_assert (FOUNDRY_IS_DOCUMENTATION_MATCHES (matches));

      if (self->stamp == stamp)
        {
          g_autoptr(GListModel) sections = foundry_documentation_matches_list_sections (matches);
          g_autoptr(GtkFlattenListModel) flatten = gtk_flatten_list_model_new (g_object_ref (sections));

          gtk_single_selection_set_model (self->search_selection, G_LIST_MODEL (flatten));
        }
    }

  return NULL;
}

static void
manuals_window_search_changed_cb (ManualsWindow  *self,
                                  GtkSearchEntry *search_entry)
{
  const char *text;

  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (GTK_IS_SEARCH_ENTRY (search_entry));

  text = gtk_editable_get_text (GTK_EDITABLE (search_entry));

  self->stamp++;

  if (text[0] == 0)
    {
      gtk_stack_set_visible_child_name (self->sidebar_stack, "browse");
      return;
    }

  gtk_stack_set_visible_child_name (self->sidebar_stack, "search");

  dex_future_disown (foundry_scheduler_spawn (NULL, 0,
                                              G_CALLBACK (manuals_window_search_fiber),
                                              3,
                                              MANUALS_TYPE_WINDOW, self,
                                              G_TYPE_STRING, text,
                                              G_TYPE_UINT, self->stamp));
}

static DexFuture *
manuals_window_search_activate_fiber (ManualsWindow        *self,
                                      FoundryDocumentation *documentation)
{
  g_autoptr(FoundryDocumentationManager) manager = NULL;
  g_autoptr(FoundryDocumentation) match = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autofree char *uri = NULL;

  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (FOUNDRY_IS_DOCUMENTATION (documentation));

  /* If we can find the same answer by doing a query for the URI
   * specifically we gain a chance of getting a documentation item
   * in the tree rather than a search result.
   *
   * This is useful so we can find siblings/parents for the navigation
   * path bar rather than just "search result".
   */

  if ((uri = foundry_documentation_dup_uri (documentation)) &&
      (context = dex_await_object (manuals_application_load_foundry (MANUALS_APPLICATION_DEFAULT), NULL)) &&
      (manager = foundry_context_dup_documentation_manager (context)) &&
      (match = dex_await_object (foundry_documentation_manager_find_by_uri (manager, uri), NULL)))
    manuals_window_navigate_to (self, match, FALSE);
  else
    manuals_window_navigate_to (self, documentation, FALSE);

  return NULL;
}

static void
manuals_window_activate_item (ManualsWindow        *self,
                              FoundryDocumentation *documentation)
{
  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (FOUNDRY_IS_DOCUMENTATION (documentation));

  dex_future_disown (foundry_scheduler_spawn (NULL, 0,
                                              G_CALLBACK (manuals_window_search_activate_fiber),
                                              2,
                                              MANUALS_TYPE_WINDOW, self,
                                              FOUNDRY_TYPE_DOCUMENTATION, documentation));
}

static void
manuals_window_search_list_activate_cb (ManualsWindow *self,
                                        guint          position,
                                        GtkListView   *list_view)
{
  g_autoptr(FoundryDocumentation) documentation = NULL;
  GListModel *model;

  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (GTK_IS_LIST_VIEW (list_view));

  model = G_LIST_MODEL (gtk_list_view_get_model (list_view));

  if ((documentation = g_list_model_get_item (model, position)))
    manuals_window_activate_item (self, documentation);
}

static void
manuals_window_search_entry_activate_cb (ManualsWindow  *self,
                                         GtkSearchEntry *search_entry)
{
  FoundryDocumentation *documentation;

  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (GTK_IS_SEARCH_ENTRY (search_entry));

  if ((documentation = gtk_single_selection_get_selected_item (self->search_selection)))
    manuals_window_activate_item (self, documentation);
}

static gboolean
manuals_window_search_entry_key_pressed_cb (ManualsWindow         *self,
                                            guint                  keyval,
                                            guint                  keycode,
                                            GdkModifierType        state,
                                            GtkEventControllerKey *controller)
{
  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (GTK_IS_EVENT_CONTROLLER_KEY (controller));

  switch (keyval)
    {
    case GDK_KEY_Down:
    case GDK_KEY_KP_Down:
      {
        GtkSelectionModel *model = gtk_list_view_get_model (self->search_list_view);
        guint selected = gtk_single_selection_get_selected (GTK_SINGLE_SELECTION (model));
        guint n_items = g_list_model_get_n_items (G_LIST_MODEL (model));

        if (n_items > 0 && selected + 1 == n_items)
          return GDK_EVENT_PROPAGATE;

        selected++;

        gtk_list_view_scroll_to (self->search_list_view, selected, GTK_LIST_SCROLL_SELECT, NULL);
      }
      return GDK_EVENT_STOP;

    case GDK_KEY_Up:
    case GDK_KEY_KP_Up:
      {
        GtkSelectionModel *model = gtk_list_view_get_model (self->search_list_view);
        guint selected = gtk_single_selection_get_selected (GTK_SINGLE_SELECTION (model));

        if (selected == 0)
          return GDK_EVENT_PROPAGATE;

        if (selected == GTK_INVALID_LIST_POSITION)
          selected = 0;
        else
          selected--;

        gtk_list_view_scroll_to (self->search_list_view, selected, GTK_LIST_SCROLL_SELECT, NULL);
      }
      return GDK_EVENT_STOP;

    case GDK_KEY_Escape:
      gtk_editable_set_text (GTK_EDITABLE (self->search_entry), "");
      return GDK_EVENT_STOP;

    case GDK_KEY_Return:
      {
        FoundryDocumentation *documentation;

        if ((documentation = gtk_single_selection_get_selected_item (self->search_selection)))
          {
            manuals_window_activate_item (self, documentation);
            return GDK_EVENT_STOP;
          }

        return GDK_EVENT_PROPAGATE;
      }

    default:
      return GDK_EVENT_PROPAGATE;
    }
}

static gboolean
nonempty_to_boolean (gpointer    instance,
                     const char *data)
{
  return data && data[0];
}

static char *
query_deprecated (gpointer              instance,
                  FoundryDocumentation *documentation)
{
  g_assert (!documentation || FOUNDRY_IS_DOCUMENTATION (documentation));

  if (documentation == NULL)
    return NULL;

  return foundry_documentation_query_attribute (documentation, FOUNDRY_DOCUMENTATION_ATTRIBUTE_DEPRECATED);
}

static char *
query_since (gpointer              instance,
             FoundryDocumentation *documentation)
{
  g_assert (!documentation || FOUNDRY_IS_DOCUMENTATION (documentation));

  if (documentation == NULL)
    return NULL;

  return foundry_documentation_query_attribute (documentation, FOUNDRY_DOCUMENTATION_ATTRIBUTE_SINCE);
}

static void
manuals_window_dispose (GObject *object)
{
  ManualsWindow *self = (ManualsWindow *)object;

  self->disposed = TRUE;

  gtk_widget_dispose_template (GTK_WIDGET (self), MANUALS_TYPE_WINDOW);

  g_clear_object (&self->visible_tab_signals);

  G_OBJECT_CLASS (manuals_window_parent_class)->dispose (object);
}

static void
manuals_window_get_property (GObject    *object,
                             guint       prop_id,
                             GValue     *value,
                             GParamSpec *pspec)
{
  ManualsWindow *self = MANUALS_WINDOW (object);

  switch (prop_id)
    {
    case PROP_VISIBLE_TAB:
      g_value_set_object (value, manuals_window_get_visible_tab (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
manuals_window_set_property (GObject      *object,
                             guint         prop_id,
                             const GValue *value,
                             GParamSpec   *pspec)
{
  ManualsWindow *self = MANUALS_WINDOW (object);

  switch (prop_id)
    {
    case PROP_VISIBLE_TAB:
      manuals_window_set_visible_tab (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
manuals_window_class_init (ManualsWindowClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->constructed = manuals_window_constructed;
  object_class->dispose = manuals_window_dispose;
  object_class->get_property = manuals_window_get_property;
  object_class->set_property = manuals_window_set_property;

  widget_class->size_allocate = manuals_window_size_allocate;

  gtk_widget_class_set_template_from_resource (widget_class, "/app/devsuite/manuals/manuals-window.ui");

  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, dock);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, list_view);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, selection);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, search_list_view);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, search_selection);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, search_entry);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, settings);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, sidebar_stack);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, split_view);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, stack);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, statusbar);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, tab_view);
  gtk_widget_class_bind_template_child (widget_class, ManualsWindow, title);

  gtk_widget_class_bind_template_callback (widget_class, on_tab_view_close_page_cb);
  gtk_widget_class_bind_template_callback (widget_class, manuals_window_list_view_activate_cb);
  gtk_widget_class_bind_template_callback (widget_class, query_deprecated);
  gtk_widget_class_bind_template_callback (widget_class, query_since);
  gtk_widget_class_bind_template_callback (widget_class, nonempty_to_boolean);
  gtk_widget_class_bind_template_callback (widget_class, manuals_window_search_changed_cb);
  gtk_widget_class_bind_template_callback (widget_class, manuals_window_search_entry_activate_cb);
  gtk_widget_class_bind_template_callback (widget_class, manuals_window_search_entry_key_pressed_cb);
  gtk_widget_class_bind_template_callback (widget_class, manuals_window_search_list_activate_cb);
  gtk_widget_class_bind_template_callback (widget_class, invert_boolean);

  gtk_widget_class_install_action (widget_class, "sidebar.focus-search", NULL, manuals_window_sidebar_focus_search_action);
  gtk_widget_class_install_action (widget_class, "tab.go-back", NULL, manuals_window_tab_go_back_action);
  gtk_widget_class_install_action (widget_class, "tab.go-forward", NULL, manuals_window_tab_go_forward_action);
  gtk_widget_class_install_action (widget_class, "tab.close", NULL, manuals_window_tab_close_action);
  gtk_widget_class_install_action (widget_class, "tab.new", NULL, manuals_window_tab_new_action);
  gtk_widget_class_install_action (widget_class, "tab.focus-search", NULL, manuals_window_tab_focus_search_action);
  gtk_widget_class_install_action (widget_class, "win.show-bundle-dialog", NULL, manuals_window_show_bundle_dialog_action);

  properties[PROP_VISIBLE_TAB] =
    g_param_spec_object ("visible-tab", NULL, NULL,
                         MANUALS_TYPE_TAB,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  g_type_ensure (MANUALS_TYPE_PATH_BAR);
  g_type_ensure (MANUALS_TYPE_SEARCH_ROW);
  g_type_ensure (MANUALS_TYPE_TAB);
  g_type_ensure (MANUALS_TYPE_TREE_EXPANDER);
  g_type_ensure (PANEL_TYPE_STATUSBAR);
}

static void
manuals_window_init (ManualsWindow *self)
{
  static const GActionEntry window_actions[] = {
    { "open-uri-in-new-tab", manuals_window_open_uri_in_new_tab_action, "s" },
    { "open-uri-in-new-window", manuals_window_open_uri_in_new_window_action, "s" },
  };

#ifdef DEVELOPMENT_BUILD
  gtk_widget_add_css_class (GTK_WIDGET (self), "devel");
#endif

  g_action_map_add_action_entries (G_ACTION_MAP (self),
                                   window_actions,
                                   G_N_ELEMENTS (window_actions),
                                   self);

  gtk_window_set_title (GTK_WINDOW (self), _("Manuals"));

  self->visible_tab_signals = g_signal_group_new (MANUALS_TYPE_TAB);
  g_signal_connect_object (self->visible_tab_signals,
                           "bind",
                           G_CALLBACK (manuals_window_update_actions),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (self->visible_tab_signals,
                           "unbind",
                           G_CALLBACK (manuals_window_update_actions),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_group_connect_object (self->visible_tab_signals,
                                 "notify::can-go-back",
                                 G_CALLBACK (manuals_window_update_actions),
                                 self,
                                 G_CONNECT_SWAPPED);
  g_signal_group_connect_object (self->visible_tab_signals,
                                 "notify::can-go-forward",
                                 G_CALLBACK (manuals_window_update_actions),
                                 self,
                                 G_CONNECT_SWAPPED);

  gtk_widget_action_set_enabled (GTK_WIDGET (self), "tab.go-back", FALSE);
  gtk_widget_action_set_enabled (GTK_WIDGET (self), "tab.go-forward", FALSE);

  gtk_widget_init_template (GTK_WIDGET (self));

  g_signal_connect_object (self->tab_view,
                           "notify::selected-page",
                           G_CALLBACK (manuals_window_tab_view_notify_selected_page_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (MANUALS_APPLICATION_DEFAULT,
                           "invalidate-contents",
                           G_CALLBACK (manuals_window_reload),
                           self,
                           G_CONNECT_SWAPPED);

  manuals_window_reload (self);
}

ManualsTab *
manuals_window_get_visible_tab (ManualsWindow *self)
{
  AdwTabPage *page;

  g_return_val_if_fail (MANUALS_IS_WINDOW (self), NULL);

  if (self->tab_view == NULL)
    return NULL;

  if ((page = adw_tab_view_get_selected_page (self->tab_view)))
    return MANUALS_TAB (adw_tab_page_get_child (page));

  return NULL;
}

void
manuals_window_set_visible_tab (ManualsWindow *self,
                                ManualsTab    *tab)
{
  AdwTabPage *page;

  g_return_if_fail (MANUALS_IS_WINDOW (self));
  g_return_if_fail (MANUALS_IS_TAB (tab));

  if ((page = adw_tab_view_get_page (self->tab_view, GTK_WIDGET (tab))))
    adw_tab_view_set_selected_page (self->tab_view, page);
}

static AdwTabPage *
manuals_window_add_tab_internal (ManualsWindow *self,
                                 ManualsTab    *tab)
{
  AdwTabPage *page;

  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (MANUALS_IS_TAB (tab));

  page = adw_tab_view_add_page (self->tab_view, GTK_WIDGET (tab), NULL);

  g_object_bind_property (tab, "title", page, "title", G_BINDING_SYNC_CREATE);
  g_object_bind_property (tab, "icon", page, "icon", G_BINDING_SYNC_CREATE);
  g_object_bind_property (tab, "loading", page, "loading", G_BINDING_SYNC_CREATE);

  manuals_window_update_stack_child (self);

  return page;
}

void
manuals_window_add_tab (ManualsWindow *self,
                        ManualsTab    *tab)
{
  g_return_if_fail (MANUALS_IS_WINDOW (self));
  g_return_if_fail (MANUALS_IS_TAB (tab));

  manuals_window_add_tab_internal (self, tab);
}

ManualsWindow *
manuals_window_new (void)
{
  return g_object_new (MANUALS_TYPE_WINDOW,
                       "application", MANUALS_APPLICATION_DEFAULT,
                       NULL);
}

ManualsWindow *
manuals_window_from_widget (GtkWidget *widget)
{
  GtkWidget *window;

  g_return_val_if_fail (GTK_IS_WIDGET (widget), NULL);

  window = gtk_widget_get_ancestor (widget, MANUALS_TYPE_WINDOW);

  g_return_val_if_fail (MANUALS_IS_WINDOW (window), NULL);

  return MANUALS_WINDOW (window);
}

static GtkTreeListRow *
get_child_row (GObject *parent,
               guint    position)
{

  if (GTK_IS_TREE_LIST_MODEL (parent))
    return gtk_tree_list_model_get_child_row (GTK_TREE_LIST_MODEL (parent), position);
  else if (GTK_IS_TREE_LIST_ROW (parent))
    return gtk_tree_list_row_get_child_row (GTK_TREE_LIST_ROW (parent), position);

  g_return_val_if_reached (NULL);
}

static GtkTreeListRow *
manuals_window_expand (GtkTreeListModel      *tree_model,
                       FoundryDocumentation **chain,
                       guint                  chain_len)
{
  g_autoptr(GObject) parent = NULL;
  gpointer rowptr;
  guint position = 0;
  guint chain_pos = 0;

  if (chain_len == 0)
    return NULL;

  parent = g_object_ref (G_OBJECT (tree_model));

  while ((rowptr = get_child_row (G_OBJECT (parent), position++)))
    {
      FoundryDocumentation *iter = chain[chain_pos];
      g_autoptr(GtkTreeListRow) row = rowptr;
      g_autoptr(FoundryDocumentation) documentation = gtk_tree_list_row_get_item (row);

      if (foundry_documentation_equal (iter, documentation))
        {
          GListModel *children;

          if (gtk_tree_list_row_is_expandable (row))
            gtk_tree_list_row_set_expanded (row, TRUE);

          g_set_object (&parent, G_OBJECT (row));
          position = 0;

          if ((children = gtk_tree_list_row_get_children (row)))
            dex_await (manuals_wrapped_model_await (MANUALS_WRAPPED_MODEL (children)), NULL);

          chain_pos++;

          if (chain_pos >= chain_len)
            return g_steal_pointer (&row);
        }
    }

  return NULL;
}

static DexFuture *
manuals_window_reveal_fiber (ManualsWindow        *self,
                             FoundryDocumentation *documentation,
                             gboolean              expand)
{
  g_autoptr(FoundryDocumentation) parent = NULL;
  g_autoptr(GtkTreeListRow) row = NULL;
  g_autoptr(GPtrArray) chain = NULL;
  GtkTreeListModel *tree_model;

  g_assert (MANUALS_IS_WINDOW (self));
  g_assert (FOUNDRY_IS_DOCUMENTATION (documentation));

  if (!(tree_model = GTK_TREE_LIST_MODEL (gtk_no_selection_get_model (self->selection))))
    return NULL;

  chain = g_ptr_array_new_with_free_func (g_object_unref);
  parent = g_object_ref (documentation);

  while (parent != NULL)
    {
      g_ptr_array_insert (chain, 0, parent);
      parent = dex_await_object (foundry_documentation_find_parent (parent), NULL);
    }

  g_ptr_array_remove_index (chain, 0);

#if 0
  for (guint i = 0; i < chain->len; i++)
    {
      g_autofree char *title = foundry_documentation_dup_title (chain->pdata[i]);
      g_message ("Tree[%u]: %s", i, title);
    }
#endif

  if ((row = manuals_window_expand (tree_model, (FoundryDocumentation **)chain->pdata, chain->len)))
    {
      guint position = gtk_tree_list_row_get_position (row);

      gtk_list_view_scroll_to (self->list_view,
                               position,
                               GTK_LIST_SCROLL_FOCUS,
                               NULL);
    }

  return NULL;
}

void
manuals_window_reveal (ManualsWindow        *self,
                       FoundryDocumentation *documentation,
                       gboolean              expand)
{
  g_return_if_fail (MANUALS_IS_WINDOW (self));
  g_return_if_fail (FOUNDRY_IS_DOCUMENTATION (documentation));

  dex_future_disown (foundry_scheduler_spawn (NULL, 0,
                                              G_CALLBACK (manuals_window_reveal_fiber),
                                              3,
                                              MANUALS_TYPE_WINDOW, self,
                                              FOUNDRY_TYPE_DOCUMENTATION, documentation,
                                              G_TYPE_BOOLEAN, expand));
}

void
manuals_window_navigate_to (ManualsWindow        *self,
                            FoundryDocumentation *navigatable,
                            gboolean              reveal)
{
  g_autofree char *uri = NULL;

  g_return_if_fail (MANUALS_IS_WINDOW (self));
  g_return_if_fail (FOUNDRY_IS_DOCUMENTATION (navigatable));

  if ((uri = foundry_documentation_dup_uri (navigatable)))
    {
      ManualsTab *tab = manuals_window_get_visible_tab (self);

      if (tab == NULL || manuals_application_control_is_pressed ())
        {
          tab = manuals_tab_new ();
          manuals_window_add_tab (self, tab);
          manuals_window_set_visible_tab (self, tab);
        }

      manuals_tab_set_navigatable (tab, navigatable);
      adw_navigation_split_view_set_show_content (self->split_view, TRUE);

      gtk_widget_grab_focus (GTK_WIDGET (tab));
    }
  else
    {
      panel_dock_set_reveal_start (self->dock, TRUE);
      adw_navigation_split_view_set_show_content (self->split_view, FALSE);
    }

  if (reveal)
    manuals_window_reveal (self, navigatable, TRUE);
}
