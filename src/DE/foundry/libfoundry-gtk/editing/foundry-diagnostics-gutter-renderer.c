/* foundry-diagnostics-gutter-renderer.c
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

#include "foundry-diagnostics-gutter-renderer.h"

struct _FoundryDiagnosticsGutterRenderer
{
  GtkSourceGutterRenderer   parent_instance;
  FoundryOnTypeDiagnostics *diagnostics;
  GtkIconPaintable         *error;
  GtkIconPaintable         *warning;
  int                       size;
  int                       width;
};

enum {
  PROP_0,
  PROP_DIAGNOSTICS,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryDiagnosticsGutterRenderer, foundry_diagnostics_gutter_renderer, GTK_SOURCE_TYPE_GUTTER_RENDERER)

static GParamSpec *properties[N_PROPS];
static GQuark error_quark;
static GQuark warning_quark;
static GIcon *error_icon;
static GIcon *warning_icon;

static void
foundry_diagnostics_gutter_renderer_snapshot_line (GtkSourceGutterRenderer *renderer,
                                                   GtkSnapshot             *snapshot,
                                                   GtkSourceGutterLines    *lines,
                                                   guint                    line)
{
  FoundryDiagnosticsGutterRenderer *self = (FoundryDiagnosticsGutterRenderer *)renderer;
  GtkIconPaintable *paintable;
  double y, h;

  g_assert (FOUNDRY_IS_DIAGNOSTICS_GUTTER_RENDERER (self));
  g_assert (GTK_IS_SNAPSHOT (snapshot));
  g_assert (lines != NULL);

  if (gtk_source_gutter_lines_has_qclass (lines, line, error_quark))
    paintable = self->error;
  else if (gtk_source_gutter_lines_has_qclass (lines, line, warning_quark))
    paintable = self->warning;
  else
    return;

  gtk_source_gutter_lines_get_line_extent (lines, line, GTK_SOURCE_GUTTER_RENDERER_ALIGNMENT_MODE_FIRST, &y, &h);

  gtk_snapshot_save (snapshot);
  gtk_snapshot_translate (snapshot,
                          &GRAPHENE_POINT_INIT ((self->width - self->size) / 2,
                                                y + ((h - self->size) / 2)));
  gdk_paintable_snapshot (GDK_PAINTABLE (paintable),
                          snapshot, self->size, self->size);
  gtk_snapshot_restore (snapshot);
}

static void
foreach_diagnostic_cb (FoundryDiagnostic *diagnostic,
                       gpointer           user_data)
{
  GtkSourceGutterLines *lines = user_data;
  guint line;

  g_assert (FOUNDRY_IS_DIAGNOSTIC (diagnostic));
  g_assert (lines != NULL);

  line = foundry_diagnostic_get_line (diagnostic);

  if (foundry_diagnostic_get_severity (diagnostic) == FOUNDRY_DIAGNOSTIC_ERROR)
    gtk_source_gutter_lines_add_qclass (lines, line, error_quark);
  else
    gtk_source_gutter_lines_add_qclass (lines, line, warning_quark);
}

static GtkIconPaintable *
get_icon (FoundryDiagnosticsGutterRenderer *self,
          GIcon                            *icon)
{
  GtkIconTheme *icon_theme;
  GdkDisplay *display;

  g_assert (FOUNDRY_IS_DIAGNOSTICS_GUTTER_RENDERER (self));
  g_assert (G_IS_ICON (icon));

  display = gtk_widget_get_display (GTK_WIDGET (self));
  icon_theme = gtk_icon_theme_get_for_display (display);

  return gtk_icon_theme_lookup_by_gicon (icon_theme,
                                         icon,
                                         gtk_widget_get_width (GTK_WIDGET (self)),
                                         gtk_widget_get_scale_factor (GTK_WIDGET (self)),
                                         gtk_widget_get_direction (GTK_WIDGET (self)),
                                         GTK_ICON_LOOKUP_FORCE_SYMBOLIC);
}

static void
foundry_diagnostics_gutter_renderer_begin (GtkSourceGutterRenderer *renderer,
                                           GtkSourceGutterLines    *lines)
{
  FoundryDiagnosticsGutterRenderer *self = (FoundryDiagnosticsGutterRenderer *)renderer;
  double y, h;
  int xpad;
  int ypad;

  g_assert (FOUNDRY_IS_DIAGNOSTICS_GUTTER_RENDERER (self));
  g_assert (lines != NULL);

  gtk_source_gutter_lines_get_line_extent (lines,
                                           gtk_source_gutter_lines_get_first (lines),
                                           GTK_SOURCE_GUTTER_RENDERER_ALIGNMENT_MODE_FIRST,
                                           &y, &h);

  xpad = gtk_source_gutter_renderer_get_xpad (renderer);
  ypad = gtk_source_gutter_renderer_get_xpad (renderer);

  self->width = gtk_widget_get_width (GTK_WIDGET (self));
  self->size = MAX (8, MIN (self->width - xpad*2, h - ypad*2));

  if (self->diagnostics == NULL)
    return;

  foundry_on_type_diagnostics_foreach_in_range (self->diagnostics,
                                                gtk_source_gutter_lines_get_first (lines),
                                                gtk_source_gutter_lines_get_last (lines),
                                                foreach_diagnostic_cb,
                                                lines);

  if (self->error == NULL)
    self->error = get_icon (self, error_icon);

  if (self->warning == NULL)
    self->warning = get_icon (self, warning_icon);
}

static void
foundry_diagnostics_gutter_renderer_dispose (GObject *object)
{
  FoundryDiagnosticsGutterRenderer *self = (FoundryDiagnosticsGutterRenderer *)object;

  g_clear_object (&self->diagnostics);
  g_clear_object (&self->error);
  g_clear_object (&self->warning);

  G_OBJECT_CLASS (foundry_diagnostics_gutter_renderer_parent_class)->dispose (object);
}

static void
foundry_diagnostics_gutter_renderer_get_property (GObject    *object,
                                                  guint       prop_id,
                                                  GValue     *value,
                                                  GParamSpec *pspec)
{
  FoundryDiagnosticsGutterRenderer *self = FOUNDRY_DIAGNOSTICS_GUTTER_RENDERER (object);

  switch (prop_id)
    {
    case PROP_DIAGNOSTICS:
      g_value_take_object (value, foundry_diagnostics_gutter_renderer_dup_diagnostics (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_diagnostics_gutter_renderer_set_property (GObject      *object,
                                                  guint         prop_id,
                                                  const GValue *value,
                                                  GParamSpec   *pspec)
{
  FoundryDiagnosticsGutterRenderer *self = FOUNDRY_DIAGNOSTICS_GUTTER_RENDERER (object);

  switch (prop_id)
    {
    case PROP_DIAGNOSTICS:
      foundry_diagnostics_gutter_renderer_set_diagnostics (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_diagnostics_gutter_renderer_class_init (FoundryDiagnosticsGutterRendererClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkSourceGutterRendererClass *renderer_class = GTK_SOURCE_GUTTER_RENDERER_CLASS (klass);

  object_class->dispose = foundry_diagnostics_gutter_renderer_dispose;
  object_class->get_property = foundry_diagnostics_gutter_renderer_get_property;
  object_class->set_property = foundry_diagnostics_gutter_renderer_set_property;

  renderer_class->begin = foundry_diagnostics_gutter_renderer_begin;
  renderer_class->snapshot_line = foundry_diagnostics_gutter_renderer_snapshot_line;

  /* TODO:
   *
   * It might be better to use GtkWidgetClass.snapshot() for this and work across
   * our known diagnostics. That could result in fewer calls to snapshot_line()
   * for a performance savings.
   *
   * We would need to store the lines from begin/end and re-use them.
   */

  properties[PROP_DIAGNOSTICS] =
    g_param_spec_object ("diagnostics", NULL, NULL,
                         FOUNDRY_TYPE_ON_TYPE_DIAGNOSTICS,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  error_quark = g_quark_from_static_string ("error");
  warning_quark = g_quark_from_static_string ("warning");

  error_icon = g_themed_icon_new ("diagnostic-error-symbolic");
  warning_icon = g_themed_icon_new ("diagnostic-warning-symbolic");
}

static void
foundry_diagnostics_gutter_renderer_init (FoundryDiagnosticsGutterRenderer *self)
{
  gtk_widget_set_size_request (GTK_WIDGET (self), 16, -1);
  gtk_source_gutter_renderer_set_xpad (GTK_SOURCE_GUTTER_RENDERER (self), 1);
}

GtkSourceGutterRenderer *
foundry_diagnostics_gutter_renderer_new (FoundryOnTypeDiagnostics *diagnostics)
{
  g_return_val_if_fail (!diagnostics || FOUNDRY_IS_ON_TYPE_DIAGNOSTICS (diagnostics), NULL);

  return g_object_new (FOUNDRY_TYPE_DIAGNOSTICS_GUTTER_RENDERER,
                       "diagnostics", diagnostics,
                       NULL);
}

/**
 * foundry_diagnostics_gutter_renderer_dup_diagnostics:
 * @self: a [class@FoundryGtk.DiagnosticsGutterRenderer]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryOnTypeDiagnostics *
foundry_diagnostics_gutter_renderer_dup_diagnostics (FoundryDiagnosticsGutterRenderer *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DIAGNOSTICS_GUTTER_RENDERER (self), NULL);

  if (self->diagnostics)
    return g_object_ref (self->diagnostics);

  return NULL;
}

void
foundry_diagnostics_gutter_renderer_set_diagnostics (FoundryDiagnosticsGutterRenderer *self,
                                                     FoundryOnTypeDiagnostics         *diagnostics)
{
  g_return_if_fail (FOUNDRY_IS_DIAGNOSTICS_GUTTER_RENDERER (self));

  if (g_set_object (&self->diagnostics, diagnostics))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_DIAGNOSTICS]);
}
