/* manuals-application.c
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

#include "manuals-application.h"
#include "manuals-model-manager.h"
#include "manuals-window.h"

struct _ManualsApplication
{
  AdwApplication  parent_instance;

  DexFuture      *foundry;
  DexFuture      *delayed_startup;

  guint           import_active : 1;
};

G_DEFINE_FINAL_TYPE (ManualsApplication, manuals_application, ADW_TYPE_APPLICATION)

enum {
  PROP_0,
  PROP_IMPORT_ACTIVE,
  N_PROPS
};

enum {
  INVALIDATE_CONTENTS,
  N_SIGNALS
};

static GParamSpec *properties[N_PROPS];
static guint signals[N_SIGNALS];

ManualsApplication *
manuals_application_new (const char        *application_id,
                         GApplicationFlags  flags)
{
  g_return_val_if_fail (application_id != NULL, NULL);

  return g_object_new (MANUALS_TYPE_APPLICATION,
                       "application-id", APP_ID,
                       "flags", flags,
                       "resource-base-path", "/app/devsuite/manuals",
                       NULL);
}

gboolean
manuals_application_control_is_pressed (void)
{
  GdkDisplay *display = gdk_display_get_default ();
  GdkSeat *seat = gdk_display_get_default_seat (display);
  GdkDevice *keyboard = gdk_seat_get_keyboard (seat);
  GdkModifierType modifiers = gdk_device_get_modifier_state (keyboard) & gtk_accelerator_get_default_mod_mask ();

  return !!(modifiers & GDK_CONTROL_MASK);
}

static DexFuture *
manuals_application_activate_cb (DexFuture *future,
                                 gpointer   user_data)
{
  ManualsWindow *window = user_data;

  g_assert (MANUALS_IS_WINDOW (window));

  gtk_window_present (GTK_WINDOW (window));

  return dex_ref (future);
}

static void
manuals_application_activate (GApplication *app)
{
  ManualsApplication *self = MANUALS_APPLICATION (app);
  GtkWindow *window;

  g_assert (MANUALS_IS_APPLICATION (self));

  if (!(window = gtk_application_get_active_window (GTK_APPLICATION (app))))
    window = GTK_WINDOW (manuals_window_new ());

  dex_future_disown (dex_future_finally (dex_ref (self->delayed_startup),
                                         manuals_application_activate_cb,
                                         g_object_ref_sink (window),
                                         g_object_unref));
}

static gint
manuals_application_command_line (GApplication            *app,
                                  GApplicationCommandLine *command_line)
{
  GVariantDict *options;
  gboolean new_window = FALSE;

  g_assert (G_IS_APPLICATION (app));
  g_assert (G_IS_APPLICATION_COMMAND_LINE (command_line));

  options = g_application_command_line_get_options_dict (command_line);

  if (g_variant_dict_lookup (options, "new-window", "b", &new_window) && new_window)
    g_action_group_activate_action (G_ACTION_GROUP (app), "new-window", NULL);
  else
    g_application_activate (app);

  return EXIT_SUCCESS;
}

static void
manuals_application_startup (GApplication *app)
{
  ManualsApplication *self = MANUALS_APPLICATION (app);
  GtkIconTheme *icon_theme;
  GdkDisplay *display;

  dex_future_disown (manuals_application_load_foundry (self));

  G_APPLICATION_CLASS (manuals_application_parent_class)->startup (app);

  display = gdk_display_get_default ();
  icon_theme = gtk_icon_theme_get_for_display (display);

  gtk_icon_theme_add_resource_path (icon_theme, "/app/devsuite/foundry/icons");
}

static void
manuals_application_shutdown (GApplication *app)
{
  ManualsApplication *self = MANUALS_APPLICATION (app);

  G_APPLICATION_CLASS (manuals_application_parent_class)->shutdown (app);

  dex_clear (&self->foundry);
  dex_clear (&self->delayed_startup);
}

static void
manuals_application_get_property (GObject    *object,
                                  guint       prop_id,
                                  GValue     *value,
                                  GParamSpec *pspec)
{
  ManualsApplication *self = MANUALS_APPLICATION (object);

  switch (prop_id)
    {
    case PROP_IMPORT_ACTIVE:
      g_value_set_boolean (value, manuals_application_get_import_active (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
manuals_application_class_init (ManualsApplicationClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GApplicationClass *app_class = G_APPLICATION_CLASS (klass);

  object_class->get_property = manuals_application_get_property;

  app_class->activate = manuals_application_activate;
  app_class->command_line = manuals_application_command_line;
  app_class->startup = manuals_application_startup;
  app_class->shutdown = manuals_application_shutdown;

  properties[PROP_IMPORT_ACTIVE] =
    g_param_spec_boolean ("import-active", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  signals[INVALIDATE_CONTENTS] =
    g_signal_new ("invalidate-contents",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 0);
}

static void
manuals_application_about_action (GSimpleAction *action,
                                  GVariant      *parameter,
                                  gpointer       user_data)
{
  static const char *developers[] = {"Christian Hergert", NULL};
  ManualsApplication *self = user_data;
  g_autoptr(GString) debug_info = NULL;
  GtkWindow *window;
  AdwDialog *dialog;

  g_assert (MANUALS_IS_APPLICATION (self));

  if (!(window = gtk_application_get_active_window (GTK_APPLICATION (self))))
    window = GTK_WINDOW (manuals_window_new ());

  debug_info = g_string_new (NULL);

  g_string_append_printf (debug_info,
                          "GTK: %u.%u.%u (Compiled against %u.%u.%u)\n",
                          gtk_get_major_version (), gtk_get_minor_version (), gtk_get_micro_version (),
                          GTK_MAJOR_VERSION, GTK_MINOR_VERSION, GTK_MICRO_VERSION);
  g_string_append_printf (debug_info,
                          "Adwaita: %u.%u.%u (Compiled against %s)\n",
                          adw_get_major_version (), adw_get_minor_version (), adw_get_micro_version (),
                          ADW_VERSION_S);
  g_string_append_printf (debug_info,
                          "Foundry: %s (Compiled against %s)\n",
                          foundry_get_version_string (),
                          FOUNDRY_VERSION_S);

  dialog = g_object_new (ADW_TYPE_ABOUT_DIALOG,
                         "application-name", _("Manuals"),
                         "application-icon", APP_ID,
                         "developer-name", "Christian Hergert",
                         "debug-info", debug_info->str,
                         "version", PACKAGE_VERSION,
                         "developers", developers,
                         "copyright", "© 2025 Christian Hergert",
                         "license-type", GTK_LICENSE_GPL_3_0,
                         "website", "https://devsuite.app/manuals",
                         "issue-url", "https://gitlab.gnome.org/chergert/manuals/issues",
                         "translator-credits", _("translator-credits"),
                         NULL);

  adw_about_dialog_add_other_app (ADW_ABOUT_DIALOG (dialog),
                                  "org.gnome.Builder",
                                  _("Builder"),
                                  _("Create GNOME Applications"));

  adw_dialog_present (dialog, GTK_WIDGET (window));
  gtk_window_present (GTK_WINDOW (window));
}

static void
manuals_application_new_window_action (GSimpleAction *action,
                                       GVariant      *parameter,
                                       gpointer       user_data)
{
  ManualsWindow *window;

  window = manuals_window_new ();
  gtk_window_present (GTK_WINDOW (window));
}

static void
manuals_application_quit_action (GSimpleAction *action,
                                 GVariant      *parameter,
                                 gpointer       user_data)
{
  ManualsApplication *self = user_data;

  g_assert (MANUALS_IS_APPLICATION (self));

  g_application_quit (G_APPLICATION (self));
}

static const GActionEntry app_actions[] = {
  { "quit", manuals_application_quit_action },
  { "about", manuals_application_about_action },
  { "new-window", manuals_application_new_window_action },
};

static const GOptionEntry main_entries[] = {
  { "new-window", 0, 0, G_OPTION_ARG_NONE, NULL, N_("Open a new Manuals window") },
  { NULL }
};

static void
manuals_application_init (ManualsApplication *self)
{
  g_autoptr(FoundryModelManager) model_manager = NULL;

  g_application_set_default (G_APPLICATION (self));
  g_application_add_main_option_entries (G_APPLICATION (self), main_entries);

  g_action_map_add_action_entries (G_ACTION_MAP (self),
                                   app_actions,
                                   G_N_ELEMENTS (app_actions),
                                   self);
  gtk_application_set_accels_for_action (GTK_APPLICATION (self),
                                         "app.quit",
                                         (const char *[]) { "<primary>q", NULL });

  model_manager = manuals_model_manager_new ();
  foundry_model_manager_set_default (model_manager);
}

static void
manuals_application_notify_indexing_cb (ManualsApplication          *self,
                                        GParamSpec                  *pspec,
                                        FoundryDocumentationManager *manager)
{
  gboolean import_active;

  g_assert (MANUALS_IS_APPLICATION (self));
  g_assert (FOUNDRY_IS_DOCUMENTATION_MANAGER (manager));

  import_active = foundry_documentation_manager_is_indexing (manager);

  if (import_active != self->import_active)
    {
      self->import_active = !!import_active;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_IMPORT_ACTIVE]);
    }
}

static DexFuture *
after_reindex (DexFuture *future,
               gpointer   user_data)
{
  ManualsApplication *self = user_data;
  g_autoptr(GError) error = NULL;

  g_assert (MANUALS_IS_APPLICATION (self));

  if (!dex_await (dex_ref (future), &error))
    g_warning ("Failed to re-index documentation: %s", error->message);

  manuals_application_reload_content (self);

  return dex_future_new_true ();
}

static void
manuals_application_documentation_changed_cb (ManualsApplication          *self,
                                              FoundryDocumentationManager *manager)
{
  g_assert (MANUALS_IS_APPLICATION (self));
  g_assert (FOUNDRY_IS_DOCUMENTATION_MANAGER (manager));

  g_debug ("Requesting that documentation be re-indexed");

  dex_future_disown (dex_future_finally (foundry_documentation_manager_index (manager),
                                         after_reindex,
                                         g_object_ref (self),
                                         g_object_unref));
}

static DexFuture *
manuals_application_track_import (DexFuture *future,
                                  gpointer   user_data)
{
  ManualsApplication *self = user_data;
  g_autoptr(FoundryContext) context = dex_await_object (dex_ref (future), NULL);
  g_autoptr(FoundryDocumentationManager) manager = foundry_context_dup_documentation_manager (context);

  g_signal_connect_object (manager,
                           "notify::indexing",
                           G_CALLBACK (manuals_application_notify_indexing_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (manager,
                           "changed",
                           G_CALLBACK (manuals_application_documentation_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  manuals_application_notify_indexing_cb (self, NULL, manager);

  return foundry_documentation_manager_index (manager);
}

DexFuture *
manuals_application_load_foundry (ManualsApplication *self)
{
  dex_return_error_if_fail (MANUALS_IS_APPLICATION (self));

  if (self->foundry == NULL)
    {
      self->foundry = foundry_context_new_for_user (NULL);
      self->delayed_startup =
        dex_future_first (dex_future_then (dex_ref (self->foundry),
                                           manuals_application_track_import,
                                           g_object_ref (self),
                                           g_object_unref),
                          dex_timeout_new_msec (500),
                          NULL);
    }

  return dex_ref (self->foundry);
}

gboolean
manuals_application_get_import_active (ManualsApplication *self)
{
  g_return_val_if_fail (MANUALS_IS_APPLICATION (self), FALSE);

  return self->import_active;
}

void
manuals_application_reload_content (ManualsApplication *self)
{
  g_assert (MANUALS_IS_APPLICATION (self));

  g_debug ("Requesting windows reload their documentation");

  g_signal_emit (self, signals[INVALIDATE_CONTENTS], 0);
}
