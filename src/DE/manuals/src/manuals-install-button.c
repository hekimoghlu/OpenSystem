/* manuals-install-button.c
 *
 * Copyright 2022 Christian Hergert <chergert@redhat.com>
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

struct _ManualsInstallButton
{
  GtkWidget         parent_instance;

  GCancellable     *cancellable;
  FoundryOperation *operation;
  DexFuture        *installing;

  GtkStack         *stack;
  GtkButton        *install;
  GtkButton        *cancel;
  GtkCssProvider   *css;

  guint             disposed : 1;
};

enum {
  PROP_0,
  PROP_LABEL,
  N_PROPS
};

enum {
  CANCEL,
  INSTALL,
  N_SIGNALS
};

G_DEFINE_FINAL_TYPE (ManualsInstallButton, manuals_install_button, GTK_TYPE_WIDGET)

static GParamSpec *properties [N_PROPS];
static guint signals [N_SIGNALS];

static void
progress_changed_cb (ManualsInstallButton *self,
                     GParamSpec           *pspec,
                     FoundryOperation     *operation)
{
  g_autofree gchar *css = NULL;
  double fraction;
  guint percentage;

  g_assert (MANUALS_IS_INSTALL_BUTTON (self));
  g_assert (FOUNDRY_IS_OPERATION (operation));

  fraction = foundry_operation_get_progress (operation);
  percentage = CLAMP (fraction * 100., .0, 100.);

  if (percentage == 0)
    css = g_strdup (".install-progress { background-size: 0; }");
  else if (percentage == 100)
    css = g_strdup (".install-progress { background-size: 100%; }");
  else
    css = g_strdup_printf (".install-progress { background-size: %u%%; }", percentage);

  gtk_css_provider_load_from_string (self->css, css);

  if (fraction >= 1.)
    {
      g_clear_object (&self->cancellable);
      g_clear_object (&self->operation);
      gtk_stack_set_visible_child (self->stack, GTK_WIDGET (self->install));
    }
}

static DexFuture *
manuals_install_button_install_fiber (gpointer data)
{
  ManualsInstallButton *self = data;
  g_autoptr(FoundryOperationManager) operation_manager = NULL;
  g_autoptr(FoundryContext) context = NULL;

  if (!(context = dex_await_object (manuals_application_load_foundry (MANUALS_APPLICATION_DEFAULT), NULL)) ||
      !(operation_manager = foundry_context_dup_operation_manager (context)) ||
      !dex_await (foundry_service_when_ready (FOUNDRY_SERVICE (operation_manager)), NULL))
    return NULL;

  g_clear_object (&self->cancellable);
  g_clear_object (&self->operation);

  self->operation = foundry_operation_manager_begin (operation_manager, _("Installing Documentation"));
  self->cancellable = g_cancellable_new ();

  g_signal_connect_object (self->operation,
                           "notify::progress",
                           G_CALLBACK (progress_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  g_signal_emit (self, signals[INSTALL], 0, self->operation, self->cancellable);

  if (self->operation != NULL)
    progress_changed_cb (self, NULL, self->operation);

  return dex_future_new_true ();
}

static void
install_clicked_cb (ManualsInstallButton *self,
                    GtkButton            *button)
{
  g_assert (MANUALS_IS_INSTALL_BUTTON (self));

  dex_clear (&self->installing);

  self->installing = dex_scheduler_spawn (NULL, 0,
                                          manuals_install_button_install_fiber,
                                          g_object_ref (self),
                                          g_object_unref);
}

static void
cancel_clicked_cb (ManualsInstallButton *self,
                   GtkButton            *button)
{
  g_assert (MANUALS_IS_INSTALL_BUTTON (self));

  g_cancellable_cancel (self->cancellable);
  dex_clear (&self->installing);

  g_signal_emit (self, signals [CANCEL], 0, self->operation, self->cancellable);
}

static void
real_install (ManualsInstallButton *self,
              FoundryOperation     *operation,
              GCancellable         *cancellable)
{
  g_assert (MANUALS_IS_INSTALL_BUTTON (self));
  g_assert (FOUNDRY_IS_OPERATION (operation));
  g_assert (G_IS_CANCELLABLE (cancellable));

  gtk_stack_set_visible_child (self->stack, GTK_WIDGET (self->cancel));
}

static void
real_cancel (ManualsInstallButton *self,
             FoundryOperation     *operation,
             GCancellable         *cancellable)
{
  g_assert (MANUALS_IS_INSTALL_BUTTON (self));
  g_assert (!operation || FOUNDRY_IS_OPERATION (operation));
  g_assert (!cancellable || G_IS_CANCELLABLE (cancellable));

  g_cancellable_cancel (cancellable);

  g_clear_object (&self->cancellable);
  g_clear_object (&self->operation);

  gtk_stack_set_visible_child (self->stack, GTK_WIDGET (self->install));
}

static void
manuals_install_button_dispose (GObject *object)
{
  ManualsInstallButton *self = (ManualsInstallButton *)object;
  GtkWidget *child;

  self->disposed = TRUE;

  g_cancellable_cancel (self->cancellable);
  g_clear_object (&self->cancellable);

  g_clear_object (&self->operation);

  dex_clear (&self->installing);

  while ((child = gtk_widget_get_first_child (GTK_WIDGET (self))))
    gtk_widget_unparent (child);

  G_OBJECT_CLASS (manuals_install_button_parent_class)->dispose (object);
}

static void
manuals_install_button_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  ManualsInstallButton *self = MANUALS_INSTALL_BUTTON (object);

  switch (prop_id)
    {
    case PROP_LABEL:
      g_value_set_string (value, manuals_install_button_get_label (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
manuals_install_button_set_property (GObject      *object,
                                     guint         prop_id,
                                     const GValue *value,
                                     GParamSpec   *pspec)
{
  ManualsInstallButton *self = MANUALS_INSTALL_BUTTON (object);

  switch (prop_id)
    {
    case PROP_LABEL:
      manuals_install_button_set_label (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
manuals_install_button_class_init (ManualsInstallButtonClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->dispose = manuals_install_button_dispose;
  object_class->get_property = manuals_install_button_get_property;
  object_class->set_property = manuals_install_button_set_property;

  properties [PROP_LABEL] =
    g_param_spec_string ("label", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_EXPLICIT_NOTIFY | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  signals [INSTALL] =
    g_signal_new_class_handler ("install",
                                G_TYPE_FROM_CLASS (klass),
                                G_SIGNAL_RUN_LAST,
                                G_CALLBACK (real_install),
                                NULL, NULL,
                                NULL,
                                G_TYPE_NONE, 2, FOUNDRY_TYPE_OPERATION, G_TYPE_CANCELLABLE);

  signals [CANCEL] =
    g_signal_new_class_handler ("cancel",
                                G_TYPE_FROM_CLASS (klass),
                                G_SIGNAL_RUN_LAST,
                                G_CALLBACK (real_cancel),
                                NULL, NULL,
                                NULL,
                                G_TYPE_NONE, 2, FOUNDRY_TYPE_OPERATION, G_TYPE_CANCELLABLE);

  gtk_widget_class_set_template_from_resource (widget_class, "/app/devsuite/manuals/manuals-install-button.ui");
  gtk_widget_class_set_layout_manager_type (widget_class, GTK_TYPE_BIN_LAYOUT);
  gtk_widget_class_bind_template_child (widget_class, ManualsInstallButton, cancel);
  gtk_widget_class_bind_template_child (widget_class, ManualsInstallButton, css);
  gtk_widget_class_bind_template_child (widget_class, ManualsInstallButton, install);
  gtk_widget_class_bind_template_child (widget_class, ManualsInstallButton, stack);
  gtk_widget_class_bind_template_callback (widget_class, install_clicked_cb);
  gtk_widget_class_bind_template_callback (widget_class, cancel_clicked_cb);
}

static void
manuals_install_button_init (ManualsInstallButton *self)
{
  gtk_widget_init_template (GTK_WIDGET (self));

  G_GNUC_BEGIN_IGNORE_DEPRECATIONS
  gtk_style_context_add_provider (gtk_widget_get_style_context (GTK_WIDGET (self->cancel)),
                                  GTK_STYLE_PROVIDER (self->css),
                                  G_MAXINT);
  G_GNUC_END_IGNORE_DEPRECATIONS
}

const char *
manuals_install_button_get_label (ManualsInstallButton *self)
{
  g_return_val_if_fail (MANUALS_IS_INSTALL_BUTTON (self), NULL);

  return gtk_button_get_label (self->install);
}

void
manuals_install_button_set_label (ManualsInstallButton *self,
                                  const char           *label)
{
  g_return_if_fail (MANUALS_IS_INSTALL_BUTTON (self));

  gtk_button_set_label (self->install, label);
  g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_LABEL]);
}

void
manuals_install_button_cancel (ManualsInstallButton *self)
{
  g_return_if_fail (MANUALS_IS_INSTALL_BUTTON (self));

  if (self->disposed)
    return;

  if (gtk_stack_get_visible_child (self->stack) == GTK_WIDGET (self->cancel))
    gtk_widget_activate (GTK_WIDGET (self->cancel));
}
