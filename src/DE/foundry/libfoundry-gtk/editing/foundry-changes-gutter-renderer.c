/* foundry-changes-gutter-renderer.c
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

#include "foundry-changes-gutter-renderer.h"
#include "foundry-source-buffer.h"
#include "foundry-source-view.h"
#include "foundry-util-private.h"

#define DELETE_HEIGHT 3
#define OVERLAP 3

struct _FoundryChangesGutterRenderer
{
  GtkSourceGutterRenderer  parent_instance;

  FoundryVcsLineChanges   *changes;
  GtkSourceGutterLines    *lines;
  DexFuture               *update_fiber;

  GdkRGBA                  added_rgba;
  GdkRGBA                  changed_rgba;
  GdkRGBA                  removed_rgba;

  guint                    show_overview : 1;
};

G_DEFINE_FINAL_TYPE (FoundryChangesGutterRenderer, foundry_changes_gutter_renderer, GTK_SOURCE_TYPE_GUTTER_RENDERER)

enum {
  PROP_0,
  PROP_SHOW_OVERVIEW,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static DexFuture *
foundry_changes_gutter_renderer_update_fiber (gpointer data)
{
  GWeakRef *wr = data;

  g_assert (wr != NULL);

  for (;;)
    {
      g_autoptr(FoundryChangesGutterRenderer) self = g_weak_ref_get (wr);
      g_autoptr(FoundryVcsLineChanges) changes = NULL;
      g_autoptr(FoundryTextDocument) document = NULL;
      g_autoptr(FoundryVcsManager) vcs_manager = NULL;
      g_autoptr(FoundryContext) context = NULL;
      g_autoptr(FoundryVcsFile) vcs_file = NULL;
      g_autoptr(FoundryVcs) vcs = NULL;
      g_autoptr(DexFuture) changed = NULL;
      g_autoptr(GBytes) contents = NULL;
      g_autoptr(GFile) file = NULL;
      GtkTextBuffer *buffer;
      GtkSourceView *view;

      if (self == NULL)
        break;

      if (!(view = gtk_source_gutter_renderer_get_view (GTK_SOURCE_GUTTER_RENDERER (self))))
        break;

      if (!FOUNDRY_IS_SOURCE_VIEW (view))
        break;

      buffer = gtk_text_view_get_buffer (GTK_TEXT_VIEW (view));
      if (!FOUNDRY_IS_SOURCE_BUFFER (buffer))
        break;

      contents = foundry_text_buffer_dup_contents (FOUNDRY_TEXT_BUFFER (buffer));
      document = foundry_source_view_dup_document (FOUNDRY_SOURCE_VIEW (view));
      file = foundry_text_document_dup_file (document);
      context = foundry_contextual_dup_context (FOUNDRY_CONTEXTUAL (document));
      vcs_manager = foundry_context_dup_vcs_manager (context);

      if (!(vcs = foundry_vcs_manager_dup_vcs (vcs_manager)))
        break;

      changed = foundry_text_document_when_changed (document);

      if (!(vcs_file = dex_await_object (foundry_vcs_find_file (vcs, file), NULL)))
        break;

      if (!(changes = dex_await_object (foundry_vcs_describe_line_changes (vcs, vcs_file, contents), NULL)))
        break;

      if (g_set_object (&self->changes, changes))
        gtk_widget_queue_draw (GTK_WIDGET (self));

      g_clear_object (&self);

      if (!dex_await (g_steal_pointer (&changed), NULL))
        break;
    }

  return dex_future_new_true ();
}

static void
foundry_changes_gutter_renderer_start (FoundryChangesGutterRenderer *self)
{
  g_assert (FOUNDRY_IS_CHANGES_GUTTER_RENDERER (self));

  if (self->update_fiber == NULL)
    self->update_fiber = dex_scheduler_spawn (NULL, 0,
                                              foundry_changes_gutter_renderer_update_fiber,
                                              foundry_weak_ref_new (self),
                                              (GDestroyNotify) foundry_weak_ref_free);
}

static void
foundry_changes_gutter_renderer_change_buffer (GtkSourceGutterRenderer *renderer,
                                               GtkSourceBuffer         *old_buffer)
{
  FoundryChangesGutterRenderer *self = (FoundryChangesGutterRenderer *)renderer;
  GtkSourceBuffer *buffer;

  g_assert (FOUNDRY_IS_CHANGES_GUTTER_RENDERER (self));
  g_assert (!old_buffer || GTK_SOURCE_IS_BUFFER (old_buffer));

  if ((buffer = gtk_source_gutter_renderer_get_buffer (renderer)))
    foundry_changes_gutter_renderer_start (self);
}

typedef struct _Snapshot
{
  GtkSnapshot          *snapshot;
  GtkSourceGutterLines *lines;
  int                   width;
  int                   height;
  guint                 n_lines;
  GdkRGBA               added;
  GdkRGBA               changed;
  GdkRGBA               removed;
  guint                 show_overview : 1;
} Snapshot;

static void
foundry_changes_gutter_renderer_snapshot_foreach (guint                line,
                                                  FoundryVcsLineChange change,
                                                  gpointer             user_data)
{
  Snapshot *state = user_data;
  double y, height;

  if (state->show_overview)
    {
      y = state->height / (double)state->n_lines * (double)line;
      height = state->height / (double)state->n_lines;
    }
  else
    {
      gtk_source_gutter_lines_get_line_extent (state->lines,
                                               line,
                                               GTK_SOURCE_GUTTER_RENDERER_ALIGNMENT_MODE_CELL,
                                               &y, &height);
    }

  if (change & FOUNDRY_VCS_LINE_ADDED)
    gtk_snapshot_append_color (state->snapshot,
                               &state->added,
                               &GRAPHENE_RECT_INIT (0, y, state->width, height));
  else if (change & FOUNDRY_VCS_LINE_CHANGED)
    gtk_snapshot_append_color (state->snapshot,
                               &state->changed,
                               &GRAPHENE_RECT_INIT (0, y, state->width, height));
  else if (change & FOUNDRY_VCS_LINE_REMOVED)
    gtk_snapshot_append_color (state->snapshot,
                               &state->removed,
                               &GRAPHENE_RECT_INIT (-OVERLAP, y, state->width + OVERLAP, DELETE_HEIGHT));
}

static void
foundry_changes_gutter_renderer_snapshot (GtkWidget   *widget,
                                          GtkSnapshot *snapshot)
{
  FoundryChangesGutterRenderer *self = (FoundryChangesGutterRenderer *)widget;
  GtkSourceBuffer *buffer;
  Snapshot state;
  guint first;
  guint last;

  g_assert (FOUNDRY_IS_CHANGES_GUTTER_RENDERER (self));
  g_assert (GTK_IS_SNAPSHOT (snapshot));

  if (self->lines == NULL || self->changes == NULL)
    return;

  if (!(buffer = gtk_source_gutter_renderer_get_buffer (GTK_SOURCE_GUTTER_RENDERER (self))))
    return;

  state.lines = self->lines;
  state.snapshot = snapshot;
  state.width = gtk_widget_get_width (widget);
  state.height = gtk_widget_get_height (widget);
  state.added = self->added_rgba;
  state.removed = self->removed_rgba;
  state.changed = self->changed_rgba;
  state.show_overview = self->show_overview;

  if (self->show_overview)
    {
      GtkTextIter iter;

      gtk_text_buffer_get_end_iter (GTK_TEXT_BUFFER (buffer), &iter);

      first = 0;
      last = gtk_text_iter_get_line (&iter);
    }
  else
    {
      first = gtk_source_gutter_lines_get_first (self->lines);
      last = gtk_source_gutter_lines_get_last (self->lines);
    }

  state.n_lines = last - first + 1;

  foundry_vcs_line_changes_foreach (self->changes, first, last,
                                    foundry_changes_gutter_renderer_snapshot_foreach,
                                    &state);
}

static void
foundry_changes_gutter_renderer_begin (GtkSourceGutterRenderer *renderer,
                                       GtkSourceGutterLines    *lines)
{
  FoundryChangesGutterRenderer *self = (FoundryChangesGutterRenderer *)renderer;

  g_assert (FOUNDRY_IS_CHANGES_GUTTER_RENDERER (self));
  g_assert (GTK_SOURCE_IS_GUTTER_LINES (lines));

  g_set_object (&self->lines, lines);
}

static void
foundry_changes_gutter_renderer_end (GtkSourceGutterRenderer *renderer)
{
  FoundryChangesGutterRenderer *self = (FoundryChangesGutterRenderer *)renderer;

  g_assert (FOUNDRY_IS_CHANGES_GUTTER_RENDERER (self));

  g_clear_object (&self->lines);
}

static void
foundry_changes_gutter_renderer_dispose (GObject *object)
{
  FoundryChangesGutterRenderer *self = (FoundryChangesGutterRenderer *)object;

  g_clear_object (&self->changes);
  g_clear_object (&self->lines);

  dex_clear (&self->update_fiber);

  G_OBJECT_CLASS (foundry_changes_gutter_renderer_parent_class)->dispose (object);
}

static void
foundry_changes_gutter_renderer_get_property (GObject    *object,
                                              guint       prop_id,
                                              GValue     *value,
                                              GParamSpec *pspec)
{
  FoundryChangesGutterRenderer *self = FOUNDRY_CHANGES_GUTTER_RENDERER (object);

  switch (prop_id)
    {
    case PROP_SHOW_OVERVIEW:
      g_value_set_boolean (value, foundry_changes_gutter_renderer_get_show_overview (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_changes_gutter_renderer_set_property (GObject      *object,
                                              guint         prop_id,
                                              const GValue *value,
                                              GParamSpec   *pspec)
{
  FoundryChangesGutterRenderer *self = FOUNDRY_CHANGES_GUTTER_RENDERER (object);

  switch (prop_id)
    {
    case PROP_SHOW_OVERVIEW:
      foundry_changes_gutter_renderer_set_show_overview (self, g_value_get_boolean (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_changes_gutter_renderer_class_init (FoundryChangesGutterRendererClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkSourceGutterRendererClass *gutter_renderer_class = GTK_SOURCE_GUTTER_RENDERER_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->dispose = foundry_changes_gutter_renderer_dispose;
  object_class->get_property = foundry_changes_gutter_renderer_get_property;
  object_class->set_property = foundry_changes_gutter_renderer_set_property;

  widget_class->snapshot = foundry_changes_gutter_renderer_snapshot;

  gutter_renderer_class->change_buffer = foundry_changes_gutter_renderer_change_buffer;
  gutter_renderer_class->begin = foundry_changes_gutter_renderer_begin;
  gutter_renderer_class->end = foundry_changes_gutter_renderer_end;

  properties[PROP_SHOW_OVERVIEW] =
    g_param_spec_boolean ("show-overview", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_changes_gutter_renderer_init (FoundryChangesGutterRenderer *self)
{
  gtk_widget_set_size_request (GTK_WIDGET (self), 2, -1);
  gtk_source_gutter_renderer_set_xpad (GTK_SOURCE_GUTTER_RENDERER (self), 0);

  /* TODO: track changes to style scheme */
  gdk_rgba_parse (&self->added_rgba, "#26a269");
  gdk_rgba_parse (&self->changed_rgba, "#f5c211");
  gdk_rgba_parse (&self->removed_rgba, "#c01c28");
}

GtkSourceGutterRenderer *
foundry_changes_gutter_renderer_new (void)
{
  return g_object_new (FOUNDRY_TYPE_CHANGES_GUTTER_RENDERER, NULL);
}

gboolean
foundry_changes_gutter_renderer_get_show_overview (FoundryChangesGutterRenderer *self)
{
  g_return_val_if_fail (FOUNDRY_IS_CHANGES_GUTTER_RENDERER (self), FALSE);

  return self->show_overview;
}

void
foundry_changes_gutter_renderer_set_show_overview (FoundryChangesGutterRenderer *self,
                                                   gboolean                      show_overview)
{
  g_return_if_fail (FOUNDRY_IS_CHANGES_GUTTER_RENDERER (self));

  show_overview = !!show_overview;

  if (show_overview != self->show_overview)
    {
      self->show_overview = show_overview;
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SHOW_OVERVIEW]);
      gtk_widget_queue_draw (GTK_WIDGET (self));
    }
}
