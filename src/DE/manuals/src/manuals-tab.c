/*
 * manuals-tab.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include <foundry.h>
#include <webkit/webkit.h>

#include <glib/gi18n.h>

#include "manuals-application.h"
#include "manuals-search-entry.h"
#include "manuals-tab.h"
#include "manuals-utils.h"
#include "manuals-window.h"

struct _ManualsTab
{
  GtkWidget             parent_instance;

  FoundryDocumentation *navigatable;

  WebKitWebView        *web_view;
  ManualsSearchEntry   *search_entry;
  GtkRevealer          *search_revealer;

  guint                 search_dir : 1;
};

G_DEFINE_FINAL_TYPE (ManualsTab, manuals_tab, GTK_TYPE_WIDGET)

enum {
  PROP_0,
  PROP_CAN_GO_BACK,
  PROP_CAN_GO_FORWARD,
  PROP_ICON,
  PROP_LOADING,
  PROP_NAVIGATABLE,
  PROP_TITLE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
manuals_tab_web_view_notify_is_loading_cb (ManualsTab *self)
{
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_LOADING]);
}

static void
manuals_tab_web_view_notify_title_cb (ManualsTab *self)
{
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_TITLE]);
}

static void
manuals_tab_web_view_notify_favicon_cb (ManualsTab *self)
{
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ICON]);
}

static void
manuals_tab_back_forward_list_changed_cb (ManualsTab *self)
{
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CAN_GO_BACK]);
  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CAN_GO_FORWARD]);
}

static ManualsWindow *
manuals_tab_get_window (ManualsTab *self)
{
  g_assert (MANUALS_IS_TAB (self));

  return MANUALS_WINDOW (gtk_widget_get_ancestor (GTK_WIDGET (self), MANUALS_TYPE_WINDOW));
}

static gboolean
manuals_tab_web_view_context_menu_cb (ManualsTab          *self,
                                      WebKitContextMenu   *context_menu,
                                      WebKitHitTestResult *hit_test_result,
                                      WebKitWebView       *web_view)
{
  GList *items;
  int i;

  g_assert (MANUALS_IS_TAB (self));
  g_assert (WEBKIT_IS_CONTEXT_MENU (context_menu));
  g_assert (WEBKIT_IS_HIT_TEST_RESULT (hit_test_result));
  g_assert (WEBKIT_IS_WEB_VIEW (web_view));

start:
  items = webkit_context_menu_get_items (context_menu);
  i = 0;
  for (; items; items = items->next)
    {
      WebKitContextMenuItem *item;
      WebKitContextMenuAction action;

      item = items->data;
      action = webkit_context_menu_item_get_stock_action (item);

      if (action == WEBKIT_CONTEXT_MENU_ACTION_DOWNLOAD_LINK_TO_DISK ||
          action == WEBKIT_CONTEXT_MENU_ACTION_DOWNLOAD_IMAGE_TO_DISK ||
          action == WEBKIT_CONTEXT_MENU_ACTION_DOWNLOAD_VIDEO_TO_DISK ||
          action == WEBKIT_CONTEXT_MENU_ACTION_DOWNLOAD_AUDIO_TO_DISK ||
          action == WEBKIT_CONTEXT_MENU_ACTION_OPEN_LINK_IN_NEW_WINDOW ||
          action == WEBKIT_CONTEXT_MENU_ACTION_OPEN_IMAGE_IN_NEW_WINDOW ||
          action == WEBKIT_CONTEXT_MENU_ACTION_OPEN_FRAME_IN_NEW_WINDOW ||
          action == WEBKIT_CONTEXT_MENU_ACTION_OPEN_VIDEO_IN_NEW_WINDOW ||
          action == WEBKIT_CONTEXT_MENU_ACTION_OPEN_AUDIO_IN_NEW_WINDOW ||
          action == WEBKIT_CONTEXT_MENU_ACTION_STOP ||
          action == WEBKIT_CONTEXT_MENU_ACTION_RELOAD)
        {
          if (action == WEBKIT_CONTEXT_MENU_ACTION_OPEN_LINK_IN_NEW_WINDOW)
            {
              WebKitContextMenuItem *new_item;
              GAction *gaction;
              const char *uri;

              uri = webkit_hit_test_result_get_link_uri (hit_test_result);

              gaction = g_action_map_lookup_action (G_ACTION_MAP (manuals_tab_get_window (self)), "open-uri-in-new-tab");
              new_item = webkit_context_menu_item_new_from_gaction (gaction,
                                                                    _("Open Link in New Tab"),
                                                                    g_variant_new_string (uri));
              webkit_context_menu_insert (context_menu, new_item, i);

              gaction = g_action_map_lookup_action (G_ACTION_MAP (manuals_tab_get_window (self)), "open-uri-in-new-window");
              new_item = webkit_context_menu_item_new_from_gaction (gaction,
                                                                    _("Open Link in New Window"),
                                                                    g_variant_new_string (uri));
              webkit_context_menu_insert (context_menu, new_item, i + 1);
            }

          webkit_context_menu_remove (context_menu, item);

          /* Start over from the beginning because we just deleted our position in the list. */
          goto start;
        }

      i++;
    }

  return GDK_EVENT_PROPAGATE;
}

typedef struct _DecidePolicy
{
  ManualsTab               *self;
  WebKitPolicyDecision     *decision;
  WebKitPolicyDecisionType  decision_type;
} DecidePolicy;

static void
decide_policy_free (DecidePolicy *state)
{
  g_clear_object (&state->self);
  g_clear_object (&state->decision);
  g_free (state);
}

static DexFuture *
manuals_tab_decide_policy_fiber (gpointer user_data)
{
  WebKitNavigationPolicyDecision *navigation_decision;
  WebKitNavigationAction *navigation_action;
  g_autoptr(FoundryDocumentationManager) doc_manager = NULL;
  g_autoptr(FoundryDocumentation) documentation = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GObject) resource = NULL;
  g_auto(GValue) uri_value = G_VALUE_INIT;
  ManualsWindow *window;
  DecidePolicy *state = user_data;
  const char *uri;
  gboolean open_new_tab = FALSE;
  int button;
  int modifiers;

  g_assert (state != NULL);
  g_assert (MANUALS_IS_TAB (state->self));
  g_assert (WEBKIT_IS_NAVIGATION_POLICY_DECISION (state->decision));

  if (!(window = manuals_tab_get_window (state->self)))
    goto ignore;

  navigation_decision = WEBKIT_NAVIGATION_POLICY_DECISION (state->decision);
  navigation_action = webkit_navigation_policy_decision_get_navigation_action (navigation_decision);
  uri = webkit_uri_request_get_uri (webkit_navigation_action_get_request (navigation_action));

  /* middle click or ctrl-click -> new tab */
  button = webkit_navigation_action_get_mouse_button (navigation_action);
  modifiers = webkit_navigation_action_get_modifiers (navigation_action);
  open_new_tab = (button == 2 || (button == 1 && modifiers == GDK_CONTROL_MASK));

  /* Pass-through API requested things */
  if (button == 0 && modifiers == 0)
    {
      webkit_policy_decision_use (state->decision);
      return dex_future_new_for_boolean (TRUE);
    }

  if (g_str_equal (uri, "about:blank"))
    {
      manuals_window_add_tab (window, manuals_tab_new ());
      goto ignore;
    }

  /* If we can find the documentation locally then use that instead of following
   * the link directly.
   */
  if ((context = dex_await_object (manuals_application_load_foundry (MANUALS_APPLICATION_DEFAULT), NULL)) &&
      (doc_manager = foundry_context_dup_documentation_manager (context)) &&
      (documentation = dex_await_object (foundry_documentation_manager_find_by_uri (doc_manager, uri), NULL)))
    {
      ManualsTab *tab = manuals_window_get_visible_tab (window);

      if (open_new_tab)
        {
          tab = manuals_tab_duplicate (tab);
          manuals_window_add_tab (window, tab);
        }

      manuals_tab_set_navigatable (tab, documentation);

      goto ignore;
    }

  /* Now if the link is not a file:/// link and we got here, defer to an
   * actual web browser rather than our internal browser.
   */
  if (g_strcmp0 ("file", g_uri_peek_scheme (uri)) != 0)
    {
      g_autoptr(GtkUriLauncher) launcher = gtk_uri_launcher_new (uri);
      gtk_uri_launcher_launch (launcher, GTK_WINDOW (window), NULL, NULL, NULL);
      goto ignore;
    }

  webkit_policy_decision_use (state->decision);

  return dex_future_new_for_boolean (TRUE);

ignore:
  webkit_policy_decision_ignore (state->decision);

  return dex_future_new_for_boolean (TRUE);
}

static gboolean
manuals_tab_web_view_decide_policy_cb (ManualsTab               *self,
                                       WebKitPolicyDecision     *decision,
                                       WebKitPolicyDecisionType  decision_type,
                                       WebKitWebView            *web_view)
{
  DecidePolicy *state;

  g_assert (MANUALS_IS_TAB (self));
  g_assert (WEBKIT_IS_POLICY_DECISION (decision));
  g_assert (WEBKIT_IS_WEB_VIEW (web_view));

  if (decision_type != WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION)
    return GDK_EVENT_PROPAGATE;

  state = g_new0 (DecidePolicy, 1);
  state->self = g_object_ref (self);
  state->decision = g_object_ref (decision);
  state->decision_type = decision_type;

  dex_future_disown (dex_scheduler_spawn (NULL, 0,
                                          manuals_tab_decide_policy_fiber,
                                          state,
                                          (GDestroyNotify)decide_policy_free));

  return GDK_EVENT_STOP;
}

static void
notify_search_revealed_cb (ManualsTab *self,
                           GParamSpec    *pspec,
                           GtkRevealer   *revealer)
{
  g_assert (MANUALS_IS_TAB (self));
  g_assert (GTK_IS_REVEALER (revealer));

  if (!gtk_revealer_get_child_revealed (revealer))
    {
      WebKitFindController *find;

      find = webkit_web_view_get_find_controller (self->web_view);
      gtk_editable_set_text (GTK_EDITABLE (self->search_entry), "");
      webkit_find_controller_search_finish (find);
    }
}

static void
search_entry_changed_cb (ManualsTab         *self,
                         GParamSpec         *pspec,
                         ManualsSearchEntry *entry)
{
  WebKitFindController *find;
  WebKitFindOptions options = 0;
  const char *text;

  g_assert (MANUALS_IS_TAB (self));
  g_assert (MANUALS_IS_SEARCH_ENTRY (entry));

  find = webkit_web_view_get_find_controller (self->web_view);
  text = gtk_editable_get_text (GTK_EDITABLE (entry));

  if (_g_str_empty0 (text))
    {
      webkit_find_controller_search_finish (find);
      manuals_search_entry_set_occurrence_count (self->search_entry, 0);
      return;
    }

  options = WEBKIT_FIND_OPTIONS_CASE_INSENSITIVE;
  options |= WEBKIT_FIND_OPTIONS_WRAP_AROUND;

  self->search_dir = 1;

  webkit_find_controller_count_matches (find, text, options, G_MAXUINT);
  webkit_find_controller_search (find, text, options, G_MAXUINT);
}

static void
search_counted_matches_cb (ManualsTab        *self,
                           guint                 match_count,
                           WebKitFindController *find)
{
  g_assert (MANUALS_IS_TAB (self));
  g_assert (WEBKIT_IS_FIND_CONTROLLER (find));

  if (match_count == G_MAXUINT)
    match_count = 0;

  manuals_search_entry_set_occurrence_position (self->search_entry, 0);
  manuals_search_entry_set_occurrence_count (self->search_entry, match_count);
}

static void
search_found_text_cb (ManualsTab        *self,
                      guint                 match_count,
                      WebKitFindController *find)
{
  int count;
  int position;

  g_assert (MANUALS_IS_TAB (self));
  g_assert (WEBKIT_IS_FIND_CONTROLLER (find));

  count = manuals_search_entry_get_occurrence_count (self->search_entry);
  position = manuals_search_entry_get_occurrence_position (self->search_entry);

  position += self->search_dir;

  if (position < 1)
    position = count;
  else if (position > count)
    position = 1;

  manuals_search_entry_set_occurrence_position (self->search_entry, position);

  gtk_widget_action_set_enabled (GTK_WIDGET (self), "search.move-next", TRUE);
  gtk_widget_action_set_enabled (GTK_WIDGET (self), "search.move-previous", TRUE);
}

static void
search_failed_to_find_text_cb (ManualsTab        *self,
                               WebKitFindController *find)
{
  g_assert (MANUALS_IS_TAB (self));
  g_assert (WEBKIT_IS_FIND_CONTROLLER (find));

  gtk_widget_action_set_enabled (GTK_WIDGET (self), "search.move-next", FALSE);
  gtk_widget_action_set_enabled (GTK_WIDGET (self), "search.move-previous", FALSE);
}

static void
search_next_action (GtkWidget  *widget,
                    const char *action_name,
                    GVariant   *param)
{
  ManualsTab *self = (ManualsTab *)widget;
  WebKitFindController *find;

  g_assert (MANUALS_IS_TAB (self));

  self->search_dir = 1;

  find = webkit_web_view_get_find_controller (self->web_view);
  webkit_find_controller_search_next (find);
}

static void
search_previous_action (GtkWidget  *widget,
                        const char *action_name,
                        GVariant   *param)
{
  ManualsTab *self = (ManualsTab *)widget;
  WebKitFindController *find;

  g_assert (MANUALS_IS_TAB (self));

  self->search_dir = -1;

  find = webkit_web_view_get_find_controller (self->web_view);
  webkit_find_controller_search_previous (find);
}

static void
hide_search_action (GtkWidget  *widget,
                    const char *action_name,
                    GVariant   *param)
{
  gtk_revealer_set_reveal_child (MANUALS_TAB (widget)->search_revealer, FALSE);
}

static void
show_search_action (GtkWidget  *widget,
                    const char *action_name,
                    GVariant   *param)
{
  gtk_revealer_set_reveal_child (MANUALS_TAB (widget)->search_revealer, TRUE);
  gtk_widget_grab_focus (GTK_WIDGET (MANUALS_TAB (widget)->search_entry));
}

static void
manuals_tab_update_background_color (ManualsTab *self)
{
  GdkRGBA background;

  g_assert (MANUALS_IS_TAB (self));

  if (adw_style_manager_get_dark (adw_style_manager_get_default ()))
    gdk_rgba_parse (&background, "#1d1d20");
  else
    gdk_rgba_parse (&background, "#ffffff");

  webkit_web_view_set_background_color (self->web_view, &background);
}

static void
manuals_tab_constructed (GObject *object)
{
  ManualsTab *self = (ManualsTab *)object;
  g_autoptr(WebKitUserStyleSheet) style_sheet = NULL;
  g_autoptr(WebKitUserScript) script = NULL;
  g_autoptr(GBytes) style_sheet_css = NULL;
  g_autoptr(GBytes) overshoot_js = NULL;
  WebKitUserContentManager *ucm;
  WebKitWebsiteDataManager *manager;
  WebKitNetworkSession *session;
  WebKitSettings *webkit_settings;
  WebKitFindController *find;

  G_OBJECT_CLASS (manuals_tab_parent_class)->constructed (object);

  webkit_settings = webkit_web_view_get_settings (self->web_view);
  webkit_settings_set_enable_back_forward_navigation_gestures (webkit_settings, TRUE);
  webkit_settings_set_enable_html5_database (webkit_settings, FALSE);
  webkit_settings_set_enable_html5_local_storage (webkit_settings, FALSE);
  webkit_settings_set_user_agent_with_application_details (webkit_settings, "GNOME-Manuals", PACKAGE_VERSION);

  ucm = webkit_web_view_get_user_content_manager (self->web_view);

  style_sheet_css = g_resources_lookup_data ("/app/devsuite/manuals/manuals-tab.css", 0, NULL);
  style_sheet = webkit_user_style_sheet_new ((const char *)g_bytes_get_data (style_sheet_css, NULL),
                                             WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES,
                                             WEBKIT_USER_STYLE_LEVEL_USER,
                                             NULL, NULL);
  webkit_user_content_manager_add_style_sheet (ucm, style_sheet);

  overshoot_js = g_resources_lookup_data ("/app/devsuite/manuals/manuals-tab.js", 0, NULL);
  script = webkit_user_script_new ((const char *)g_bytes_get_data (overshoot_js, NULL),
                                   WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES,
                                   WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_END,
                                   NULL, NULL);
  webkit_user_content_manager_add_script (ucm, script);

  session = webkit_web_view_get_network_session (self->web_view);
  manager = webkit_network_session_get_website_data_manager (session);
  webkit_website_data_manager_set_favicons_enabled (manager, TRUE);

  find = webkit_web_view_get_find_controller (self->web_view);
  g_signal_connect_object (find,
                           "counted-matches",
                           G_CALLBACK (search_counted_matches_cb),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (find,
                           "found-text",
                           G_CALLBACK (search_found_text_cb),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (find,
                           "failed-to-find-text",
                           G_CALLBACK (search_failed_to_find_text_cb),
                           self,
                           G_CONNECT_SWAPPED);

  manuals_tab_update_background_color (self);
}

static void
manuals_tab_css_changed (GtkWidget         *widget,
                         GtkCssStyleChange *change)
{
  manuals_tab_update_background_color (MANUALS_TAB (widget));
}

static void
manuals_tab_dispose (GObject *object)
{
  ManualsTab *self = (ManualsTab *)object;
  GtkWidget *child;

  gtk_widget_dispose_template (GTK_WIDGET (self), MANUALS_TYPE_TAB);

  while ((child = gtk_widget_get_first_child (GTK_WIDGET (self))))
    gtk_widget_unparent (child);

  g_clear_object (&self->navigatable);

  G_OBJECT_CLASS (manuals_tab_parent_class)->dispose (object);
}

static void
manuals_tab_get_property (GObject    *object,
                          guint       prop_id,
                          GValue     *value,
                          GParamSpec *pspec)
{
  ManualsTab *self = MANUALS_TAB (object);

  switch (prop_id)
    {
    case PROP_CAN_GO_BACK:
      g_value_set_boolean (value, manuals_tab_can_go_back (self));
      break;

    case PROP_CAN_GO_FORWARD:
      g_value_set_boolean (value, manuals_tab_can_go_forward (self));
      break;

    case PROP_ICON:
      g_value_take_object (value, manuals_tab_dup_icon (self));
      break;

    case PROP_LOADING:
      g_value_set_boolean (value, manuals_tab_get_loading (self));
      break;

    case PROP_NAVIGATABLE:
      g_value_set_object (value, manuals_tab_get_navigatable (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, manuals_tab_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
manuals_tab_set_property (GObject      *object,
                          guint         prop_id,
                          const GValue *value,
                          GParamSpec   *pspec)
{
  ManualsTab *self = MANUALS_TAB (object);

  switch (prop_id)
    {
    case PROP_NAVIGATABLE:
      manuals_tab_set_navigatable (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
manuals_tab_class_init (ManualsTabClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->constructed = manuals_tab_constructed;
  object_class->dispose = manuals_tab_dispose;
  object_class->get_property = manuals_tab_get_property;
  object_class->set_property = manuals_tab_set_property;

  widget_class->css_changed = manuals_tab_css_changed;

  properties[PROP_CAN_GO_BACK] =
    g_param_spec_boolean ("can-go-back", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_CAN_GO_FORWARD] =
    g_param_spec_boolean ("can-go-forward", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_ICON] =
    g_param_spec_object ("icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_LOADING] =
    g_param_spec_boolean ("loading", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_NAVIGATABLE] =
    g_param_spec_object ("navigatable", NULL, NULL,
                         FOUNDRY_TYPE_DOCUMENTATION,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  gtk_widget_class_set_template_from_resource (widget_class, "/app/devsuite/manuals/manuals-tab.ui");
  gtk_widget_class_set_layout_manager_type (widget_class, GTK_TYPE_BIN_LAYOUT);

  gtk_widget_class_bind_template_child (widget_class, ManualsTab, web_view);
  gtk_widget_class_bind_template_child (widget_class, ManualsTab, search_entry);
  gtk_widget_class_bind_template_child (widget_class, ManualsTab, search_revealer);

  gtk_widget_class_bind_template_callback (widget_class, search_entry_changed_cb);
  gtk_widget_class_bind_template_callback (widget_class, notify_search_revealed_cb);

  gtk_widget_class_install_action (widget_class, "search.hide", NULL, hide_search_action);
  gtk_widget_class_install_action (widget_class, "search.show", NULL, show_search_action);
  gtk_widget_class_install_action (widget_class, "search.move-next", NULL, search_next_action);
  gtk_widget_class_install_action (widget_class, "search.move-previous", NULL, search_previous_action);

  g_type_ensure (MANUALS_TYPE_SEARCH_ENTRY);
  g_type_ensure (WEBKIT_TYPE_WEB_VIEW);
}

static void
manuals_tab_init (ManualsTab *self)
{
  WebKitBackForwardList *back_forward_list;

  gtk_widget_init_template (GTK_WIDGET (self));

  back_forward_list = webkit_web_view_get_back_forward_list (self->web_view);

  g_signal_connect_object (self->web_view,
                           "decide-policy",
                           G_CALLBACK (manuals_tab_web_view_decide_policy_cb),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (self->web_view,
                           "notify::is-loading",
                           G_CALLBACK (manuals_tab_web_view_notify_is_loading_cb),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (self->web_view,
                           "notify::favicon",
                           G_CALLBACK (manuals_tab_web_view_notify_favicon_cb),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (self->web_view,
                           "notify::title",
                           G_CALLBACK (manuals_tab_web_view_notify_title_cb),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (self->web_view,
                           "context-menu",
                           G_CALLBACK (manuals_tab_web_view_context_menu_cb),
                           self,
                           G_CONNECT_SWAPPED);
  g_signal_connect_object (back_forward_list,
                           "changed",
                           G_CALLBACK (manuals_tab_back_forward_list_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);
}

ManualsTab *
manuals_tab_new (void)
{
  return g_object_new (MANUALS_TYPE_TAB, NULL);
}

ManualsTab *
manuals_tab_duplicate (ManualsTab *self)
{
  ManualsTab *copy;

  g_return_val_if_fail (!self || MANUALS_IS_TAB (self), NULL);

  copy = g_object_new (MANUALS_TYPE_TAB, NULL);

  if (self != NULL)
    {
      g_autoptr(WebKitWebViewSessionState) state = NULL;

      manuals_tab_set_navigatable (copy, self->navigatable);

      /* Create the new tab using the back/forward list of the original tab. */
      state = webkit_web_view_get_session_state (self->web_view);
      webkit_web_view_restore_session_state (copy->web_view, state);
    }

  return copy;
}

char *
manuals_tab_dup_title (ManualsTab *self)
{
  g_autofree char *tmp = NULL;
  const char *title;

  g_return_val_if_fail (MANUALS_IS_TAB (self), NULL);

  title = webkit_web_view_get_title (self->web_view);

  if (_g_str_empty0 (title) && self->navigatable != NULL)
    title = tmp = foundry_documentation_dup_title (self->navigatable);

  if (_g_str_empty0 (title))
    title = _("Empty Page");

  return g_strdup (title);
}


GIcon *
manuals_tab_dup_icon (ManualsTab *self)
{
  GdkTexture *texture;

  g_return_val_if_fail (MANUALS_IS_TAB (self), NULL);

  if ((texture = webkit_web_view_get_favicon (self->web_view)))
    return g_object_ref (G_ICON (texture));

  return NULL;
}

gboolean
manuals_tab_get_loading (ManualsTab *self)
{
  g_return_val_if_fail (MANUALS_IS_TAB (self), FALSE);

  return webkit_web_view_is_loading (self->web_view);
}

gboolean
manuals_tab_can_go_back (ManualsTab *self)
{
  g_return_val_if_fail (MANUALS_IS_TAB (self), FALSE);

  return webkit_web_view_can_go_back (self->web_view);
}

gboolean
manuals_tab_can_go_forward (ManualsTab *self)
{
  g_return_val_if_fail (MANUALS_IS_TAB (self), FALSE);

  return webkit_web_view_can_go_forward (self->web_view);
}

void
manuals_tab_go_back (ManualsTab *self)
{
  g_return_if_fail (MANUALS_IS_TAB (self));

  webkit_web_view_go_back (self->web_view);
}

void
manuals_tab_go_forward (ManualsTab *self)
{
  g_return_if_fail (MANUALS_IS_TAB (self));

  webkit_web_view_go_forward (self->web_view);
}

FoundryDocumentation *
manuals_tab_get_navigatable (ManualsTab *self)
{
  g_return_val_if_fail (MANUALS_IS_TAB (self), NULL);

  return self->navigatable;
}

void
manuals_tab_set_navigatable (ManualsTab           *self,
                             FoundryDocumentation *navigatable)
{
  g_return_if_fail (MANUALS_IS_TAB (self));
  g_return_if_fail (!navigatable || FOUNDRY_IS_DOCUMENTATION (navigatable));

  if (g_set_object (&self->navigatable, navigatable))
    {
      const char *uri = NULL;

      if (navigatable != NULL)
        uri = foundry_documentation_dup_uri (navigatable);

      if (uri != NULL)
        webkit_web_view_load_uri (self->web_view, uri);

      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_NAVIGATABLE]);
    }
}

void
manuals_tab_load_uri (ManualsTab *self,
                      const char *uri)
{
  g_return_if_fail (MANUALS_IS_TAB (self));
  g_return_if_fail (uri != NULL);

  if (g_set_object (&self->navigatable, NULL))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_NAVIGATABLE]);

  webkit_web_view_load_uri (self->web_view, uri);
}

void
manuals_tab_focus_search (ManualsTab *self)
{
  g_return_if_fail (MANUALS_IS_TAB (self));

  gtk_widget_activate_action (GTK_WIDGET (self), "search.show", NULL);
}
