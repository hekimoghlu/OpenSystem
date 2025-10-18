/*
 * manuals-bundle-dialog.c
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

#include <glib/gi18n.h>

#include <foundry.h>

#include "manuals-application.h"
#include "manuals-install-button.h"
#include "manuals-bundle-dialog.h"
#include "manuals-tag.h"

struct _ManualsBundleDialog
{
  AdwPreferencesDialog  parent_instance;

  GtkFilterListModel   *installed;
  AdwPreferencesGroup  *installed_group;
  GtkListBox           *installed_list_box;
  GtkFilterListModel   *available;
  AdwPreferencesGroup  *available_group;
  GtkListBox           *available_list_box;
  GtkStack             *stack;
};

G_DEFINE_FINAL_TYPE (ManualsBundleDialog, manuals_bundle_dialog, ADW_TYPE_PREFERENCES_DIALOG)

static void manuals_bundle_dialog_reload (ManualsBundleDialog *self);

static DexFuture *
manuals_bundle_dialog_cancel (DexFuture *completed,
                              gpointer   user_data)
{
  manuals_install_button_cancel (MANUALS_INSTALL_BUTTON (user_data));
  return NULL;
}

static DexFuture *
manuals_bundle_dialog_ensure_done (DexFuture *completed,
                                   gpointer   user_data)
{
  foundry_operation_complete (FOUNDRY_OPERATION (user_data));
  manuals_application_reload_content (MANUALS_APPLICATION_DEFAULT);
  return NULL;
}

static void
manuals_bundle_dialog_install_cb (FoundryDocumentationBundle *bundle,
                                  FoundryOperation           *operation,
                                  GCancellable               *cancellable,
                                  ManualsInstallButton       *button)
{
  g_autoptr(DexCancellable) cancel = NULL;
  DexFuture *future;

  g_assert (FOUNDRY_IS_DOCUMENTATION_BUNDLE (bundle));
  g_assert (FOUNDRY_IS_OPERATION (operation));
  g_assert (!cancellable || G_IS_CANCELLABLE (cancellable));
  g_assert (MANUALS_IS_INSTALL_BUTTON (button));

  cancel = DEX_CANCELLABLE (dex_cancellable_new_from_cancellable (cancellable));

  future = foundry_documentation_bundle_install (bundle, operation, cancel);
  future = dex_future_finally (future,
                               manuals_bundle_dialog_cancel,
                               g_object_ref (button),
                               g_object_unref);
  future = dex_future_finally (future,
                               manuals_bundle_dialog_ensure_done,
                               g_object_ref (operation),
                               g_object_unref);

  dex_future_disown (future);
}

static void
manuals_bundle_dialog_cancel_cb (FoundryDocumentationBundle *bundle,
                                 FoundryOperation           *operation,
                                 GCancellable               *cancellable,
                                 ManualsInstallButton       *button)
{
  g_assert (FOUNDRY_IS_DOCUMENTATION_BUNDLE (bundle));
  g_assert (!operation || FOUNDRY_IS_OPERATION (operation));
  g_assert (!cancellable || G_IS_CANCELLABLE (cancellable));
  g_assert (MANUALS_IS_INSTALL_BUTTON (button));

  g_cancellable_cancel (cancellable);
}

static GtkWidget *
create_bundle_row (gpointer item,
                   gpointer user_data)
{
  FoundryDocumentationBundle *bundle = item;
  g_autofree char *title = NULL;
  g_autofree char *subtitle = NULL;
  g_auto(GStrv) tags = NULL;
  GtkWidget *row;

  g_assert (FOUNDRY_IS_DOCUMENTATION_BUNDLE (bundle));
  g_assert (MANUALS_IS_BUNDLE_DIALOG (user_data));

  title = foundry_documentation_bundle_dup_title (bundle);
  subtitle = foundry_documentation_bundle_dup_subtitle (bundle);
  tags = foundry_documentation_bundle_dup_tags (bundle);

  row = g_object_new (ADW_TYPE_ACTION_ROW,
                      "title", title,
                      "subtitle", subtitle,
                      NULL);

  if (tags != NULL)
    {
      for (guint i = 0; tags[i]; i++)
        adw_action_row_add_suffix (ADW_ACTION_ROW (row),
                                   g_object_new (MANUALS_TYPE_TAG,
                                                 "css-classes", (const char * const []) { "installation", NULL },
                                                 "value", tags[i],
                                                 "valign", GTK_ALIGN_CENTER,
                                                 NULL));
    }

  if (!foundry_documentation_bundle_get_installed (bundle))
    {
      GtkWidget *button;

      button = g_object_new (MANUALS_TYPE_INSTALL_BUTTON,
                             "label", _("Install"),
                             "valign", GTK_ALIGN_CENTER,
                             NULL);
      g_signal_connect_object (button,
                               "install",
                               G_CALLBACK (manuals_bundle_dialog_install_cb),
                               bundle,
                               G_CONNECT_SWAPPED);
      g_signal_connect_object (button,
                               "cancel",
                               G_CALLBACK (manuals_bundle_dialog_cancel_cb),
                               bundle,
                               G_CONNECT_SWAPPED);

      adw_action_row_add_suffix (ADW_ACTION_ROW (row), button);
      adw_action_row_set_activatable_widget (ADW_ACTION_ROW (row), button);
    }

  g_object_set_data_full (G_OBJECT (row),
                          "FOUNDRY_DOCUMENTATION_BUNDLE",
                          g_object_ref (bundle),
                          g_object_unref);

  return row;
}

static int
ref_sorter (gconstpointer a,
            gconstpointer b,
            gpointer      user_data)
{
  FoundryDocumentationBundle *ref_a = (gpointer)a;
  FoundryDocumentationBundle *ref_b = (gpointer)b;
  g_autofree char *title_a = foundry_documentation_bundle_dup_title (ref_a);
  g_autofree char *title_b = foundry_documentation_bundle_dup_title (ref_b);
  gboolean a_is_gnome = strstr (title_a, "GNOME") != NULL;
  gboolean b_is_gnome = strstr (title_b, "GNOME") != NULL;
  int ret;

  if (a_is_gnome && !b_is_gnome)
    return -1;

  if (!a_is_gnome && b_is_gnome)
    return 1;

  ret = g_strcmp0 (title_a, title_b);

  /* Reverse sort */
  if (ret < 0)
    return 1;
  else if (ret > 1)
    return -1;
  else
    return 0;
}

static void
installed_items_changed_cb (ManualsBundleDialog *self,
                            guint                position,
                            guint                removed,
                            guint                added,
                            GListModel          *model)
{
  guint n_items;

  g_assert (MANUALS_IS_BUNDLE_DIALOG (self));
  g_assert (G_IS_LIST_MODEL (model));

  n_items = g_list_model_get_n_items (model);

  gtk_widget_set_visible (GTK_WIDGET (self->installed_group), !!n_items);
}

static void
available_items_changed_cb (ManualsBundleDialog *self,
                            guint             position,
                            guint             removed,
                            guint             added,
                            GListModel       *model)
{
  guint n_items;

  g_assert (MANUALS_IS_BUNDLE_DIALOG (self));
  g_assert (G_IS_LIST_MODEL (model));

  n_items = g_list_model_get_n_items (model);

  gtk_widget_set_visible (GTK_WIDGET (self->available_group), !!n_items);
}

static void
manuals_bundle_dialog_dispose (GObject *object)
{
  ManualsBundleDialog *self = (ManualsBundleDialog *)object;

  gtk_widget_dispose_template (GTK_WIDGET (self), MANUALS_TYPE_BUNDLE_DIALOG);

  G_OBJECT_CLASS (manuals_bundle_dialog_parent_class)->dispose (object);
}

static void
manuals_bundle_dialog_class_init (ManualsBundleDialogClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->dispose = manuals_bundle_dialog_dispose;

  gtk_widget_class_set_template_from_resource (widget_class, "/app/devsuite/manuals/manuals-bundle-dialog.ui");

  gtk_widget_class_bind_template_child (widget_class, ManualsBundleDialog, available);
  gtk_widget_class_bind_template_child (widget_class, ManualsBundleDialog, available_group);
  gtk_widget_class_bind_template_child (widget_class, ManualsBundleDialog, available_list_box);
  gtk_widget_class_bind_template_child (widget_class, ManualsBundleDialog, installed);
  gtk_widget_class_bind_template_child (widget_class, ManualsBundleDialog, installed_group);
  gtk_widget_class_bind_template_child (widget_class, ManualsBundleDialog, installed_list_box);
  gtk_widget_class_bind_template_child (widget_class, ManualsBundleDialog, stack);

  g_type_ensure (FOUNDRY_TYPE_DOCUMENTATION_BUNDLE);
}

static void
manuals_bundle_dialog_init (ManualsBundleDialog *self)
{
  gtk_widget_init_template (GTK_WIDGET (self));

  g_signal_connect_object (self->installed,
                           "items-changed",
                           G_CALLBACK (installed_items_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (self->available,
                           "items-changed",
                           G_CALLBACK (available_items_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_connect_object (MANUALS_APPLICATION_DEFAULT,
                           "invalidate-contents",
                           G_CALLBACK (manuals_bundle_dialog_reload),
                           self,
                           G_CONNECT_SWAPPED);
}

ManualsBundleDialog *
manuals_bundle_dialog_new (void)
{
  return g_object_new (MANUALS_TYPE_BUNDLE_DIALOG, NULL);
}

static DexFuture *
manuals_bundle_dialog_present_fiber (gpointer user_data)
{
  ManualsBundleDialog *self = user_data;
  g_autoptr(FoundryDocumentationManager) documentation_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(GListModel) bundles = NULL;
  g_autoptr(GtkSortListModel) sorted = NULL;
  g_autoptr(GtkCustomSorter) sorter = NULL;

  g_assert (MANUALS_IS_BUNDLE_DIALOG (self));

  if (!(context = dex_await_object (manuals_application_load_foundry (MANUALS_APPLICATION_DEFAULT), NULL)) ||
      !(documentation_manager = foundry_context_dup_documentation_manager (context)) ||
      !dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (documentation_manager)), NULL) ||
      !(bundles = dex_await_object (foundry_documentation_manager_list_bundles (documentation_manager), NULL)) ||
      g_list_model_get_n_items (bundles) == 0)
    {
      gtk_stack_set_visible_child_name (self->stack, "empty");
      return NULL;
    }

  gtk_stack_set_visible_child_name (self->stack, "list");

  sorted = gtk_sort_list_model_new (NULL, NULL);
  sorter = gtk_custom_sorter_new (ref_sorter, NULL, NULL);
  gtk_sort_list_model_set_model (sorted, G_LIST_MODEL (bundles));
  gtk_sort_list_model_set_sorter (sorted, GTK_SORTER (sorter));

  gtk_filter_list_model_set_model (self->installed, G_LIST_MODEL (sorted));
  gtk_filter_list_model_set_model (self->available, G_LIST_MODEL (sorted));

  gtk_list_box_bind_model (self->installed_list_box,
                           G_LIST_MODEL (self->installed),
                           create_bundle_row, self, NULL);
  gtk_list_box_bind_model (self->available_list_box,
                           G_LIST_MODEL (self->available),
                           create_bundle_row, self, NULL);

  installed_items_changed_cb (self, 0, 0, 0, G_LIST_MODEL (self->installed));
  available_items_changed_cb (self, 0, 0, 0, G_LIST_MODEL (self->available));

  return NULL;
}

static void
manuals_bundle_dialog_reload (ManualsBundleDialog *self)
{
  g_return_if_fail (MANUALS_IS_BUNDLE_DIALOG (self));

  dex_future_disown (dex_scheduler_spawn (NULL, 0,
                                          manuals_bundle_dialog_present_fiber,
                                          g_object_ref (self),
                                          g_object_unref));
}

void
manuals_bundle_dialog_present (ManualsBundleDialog *self,
                               GtkWidget           *parent)
{
  g_return_if_fail (MANUALS_IS_BUNDLE_DIALOG (self));
  g_return_if_fail (GTK_IS_WIDGET (parent));

  manuals_bundle_dialog_reload (self);

  adw_dialog_present (ADW_DIALOG (self), parent);
}
