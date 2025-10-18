/* plugin-spellcheck-source-view-addin.c
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

#include "plugin-spellcheck-text-document-addin.h"
#include "plugin-spellcheck-source-view-addin.h"

struct _PluginSpellcheckSourceViewAddin
{
  FoundrySourceViewAddin  parent_instance;
  GtkGesture             *pressed;
  GMenuModel             *menu;
};

G_DEFINE_FINAL_TYPE (PluginSpellcheckSourceViewAddin, plugin_spellcheck_source_view_addin, FOUNDRY_TYPE_SOURCE_VIEW_ADDIN)

static void
on_click_pressed_cb (PluginSpellcheckSourceViewAddin *self,
                     int                              n_press,
                     double                           x,
                     double                           y,
                     GtkGestureClick                 *click)
{
  g_autoptr(PluginSpellcheckTextDocumentAddin) addin = NULL;
  g_autoptr(FoundryTextDocument) document = NULL;
  FoundrySourceView *view;
  GdkEventSequence *sequence;
  GtkTextBuffer *buffer;
  GtkTextIter iter, begin, end;
  GtkWidget *widget;
  GdkEvent *event;
  int buf_x, buf_y;

  g_assert (PLUGIN_IS_SPELLCHECK_SOURCE_VIEW_ADDIN (self));
  g_assert (GTK_IS_GESTURE_CLICK (click));

  widget = gtk_event_controller_get_widget (GTK_EVENT_CONTROLLER (click));
  sequence = gtk_gesture_single_get_current_sequence (GTK_GESTURE_SINGLE (click));
  event = gtk_gesture_get_last_event (GTK_GESTURE (click), sequence);

  if (n_press != 1 || !gdk_event_triggers_context_menu (event))
    return;

  buffer = gtk_text_view_get_buffer (GTK_TEXT_VIEW (widget));
  if (gtk_text_buffer_get_selection_bounds (buffer, &begin, &end))
    return;

  gtk_text_view_window_to_buffer_coords (GTK_TEXT_VIEW (widget),
                                         GTK_TEXT_WINDOW_WIDGET,
                                         x, y, &buf_x, &buf_y);
  gtk_text_view_get_iter_at_location (GTK_TEXT_VIEW (widget), &iter, buf_x, buf_y);
  gtk_text_buffer_select_range (buffer, &iter, &iter);

  view = foundry_source_view_addin_get_view (FOUNDRY_SOURCE_VIEW_ADDIN (self));
  document = foundry_source_view_dup_document (view);
  addin = foundry_text_document_find_addin (document, "spellcheck");

  plugin_spellcheck_text_document_addin_update_corrections (addin);
}

static DexFuture *
plugin_spellcheck_source_view_addin_load (FoundrySourceViewAddin *addin)
{
  PluginSpellcheckSourceViewAddin *self = (PluginSpellcheckSourceViewAddin *)addin;
  g_autoptr(PluginSpellcheckTextDocumentAddin) document_addin = NULL;
  g_autoptr(FoundryTextDocument) document = NULL;
  FoundrySourceView *view;
  GMenuModel *menu;

  g_assert (PLUGIN_IS_SPELLCHECK_SOURCE_VIEW_ADDIN (self));

  view = foundry_source_view_addin_get_view (addin);
  document = foundry_source_view_dup_document (view);
  document_addin = foundry_text_document_find_addin (document, "spellcheck");
  menu = plugin_spellcheck_text_document_addin_get_menu (document_addin);

  if (g_set_object (&self->menu, menu))
    foundry_source_view_append_menu (view, menu);

  self->pressed = gtk_gesture_click_new ();
  gtk_gesture_single_set_button (GTK_GESTURE_SINGLE (self->pressed), 0);
  gtk_event_controller_set_propagation_phase (GTK_EVENT_CONTROLLER (self->pressed), GTK_PHASE_CAPTURE);
  g_signal_connect_object (self->pressed,
                           "pressed",
                           G_CALLBACK (on_click_pressed_cb),
                           self,
                           G_CONNECT_SWAPPED);
  gtk_widget_add_controller (GTK_WIDGET (view),
                             g_object_ref (GTK_EVENT_CONTROLLER (self->pressed)));

  return dex_future_new_true ();
}

static DexFuture *
plugin_spellcheck_source_view_addin_unload (FoundrySourceViewAddin *addin)
{
  PluginSpellcheckSourceViewAddin *self = (PluginSpellcheckSourceViewAddin *)addin;
  FoundrySourceView *view;

  g_assert (PLUGIN_IS_SPELLCHECK_SOURCE_VIEW_ADDIN (self));

  view = foundry_source_view_addin_get_view (addin);
  gtk_widget_remove_controller (GTK_WIDGET (view), GTK_EVENT_CONTROLLER (self->pressed));
  g_clear_object (&self->pressed);

  if (self->menu != NULL)
    {
      foundry_source_view_remove_menu (view, self->menu);
      g_clear_object (&self->menu);
    }

  return dex_future_new_true ();
}

static void
plugin_spellcheck_source_view_addin_class_init (PluginSpellcheckSourceViewAddinClass *klass)
{
  FoundrySourceViewAddinClass *addin_class = FOUNDRY_SOURCE_VIEW_ADDIN_CLASS (klass);

  addin_class->load = plugin_spellcheck_source_view_addin_load;
  addin_class->unload = plugin_spellcheck_source_view_addin_unload;
}

static void
plugin_spellcheck_source_view_addin_init (PluginSpellcheckSourceViewAddin *self)
{
}
