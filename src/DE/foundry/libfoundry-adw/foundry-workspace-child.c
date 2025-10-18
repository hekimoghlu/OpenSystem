/* foundry-workspace-child.c
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

#include <libpanel.h>

#include "foundry-page.h"
#include "foundry-workspace-child-private.h"

struct _FoundryWorkspaceChild
{
  GObject                    parent_instance;
  GBindingGroup             *bindings;
  GtkWidget                 *child;
  GIcon                     *icon;
  char                      *subtitle;
  char                      *title;
  AdwBin                    *narrow_widget;
  PanelWidget               *wide_widget;
  FoundryWorkspaceChildKind  kind : 1;
  guint                      modified : 1;
  guint                      area : 3;
};

G_DEFINE_FINAL_TYPE (FoundryWorkspaceChild, foundry_workspace_child, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_AREA,
  PROP_CHILD,
  PROP_ICON,
  PROP_KIND,
  PROP_MODIFIED,
  PROP_SUBTITLE,
  PROP_TITLE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

FoundryWorkspaceChild *
foundry_workspace_child_new (FoundryWorkspaceChildKind kind,
                             PanelArea                 area)
{
  g_return_val_if_fail (kind == FOUNDRY_WORKSPACE_CHILD_PAGE ||
                        kind == FOUNDRY_WORKSPACE_CHILD_PANEL, NULL);

  return g_object_new (FOUNDRY_TYPE_WORKSPACE_CHILD,
                       "kind", kind,
                       "area", area,
                       NULL);
}

static void
foundry_workspace_child_dispose (GObject *object)
{
  FoundryWorkspaceChild *self = (FoundryWorkspaceChild *)object;

  if (self->bindings)
    g_binding_group_set_source (self->bindings, NULL);

  if (self->narrow_widget)
    adw_bin_set_child (self->narrow_widget, NULL);

  if (self->wide_widget)
    panel_widget_set_child (self->wide_widget, NULL);

  g_clear_pointer (&self->title, g_free);
  g_clear_pointer (&self->subtitle, g_free);

  g_clear_object (&self->bindings);
  g_clear_object (&self->child);
  g_clear_object (&self->icon);
  g_clear_object (&self->narrow_widget);
  g_clear_object (&self->wide_widget);

  G_OBJECT_CLASS (foundry_workspace_child_parent_class)->dispose (object);
}

static void
foundry_workspace_child_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  FoundryWorkspaceChild *self = FOUNDRY_WORKSPACE_CHILD (object);

  switch (prop_id)
    {
    case PROP_AREA:
      g_value_set_enum (value, foundry_workspace_child_get_area (self));
      break;

    case PROP_CHILD:
      g_value_set_object (value, foundry_workspace_child_get_child (self));
      break;

    case PROP_ICON:
      g_value_set_object (value, foundry_workspace_child_get_icon (self));
      break;

    case PROP_KIND:
      g_value_set_enum (value, foundry_workspace_child_get_kind (self));
      break;

    case PROP_MODIFIED:
      g_value_set_boolean (value, foundry_workspace_child_get_modified (self));
      break;

    case PROP_SUBTITLE:
      g_value_set_string (value, foundry_workspace_child_get_subtitle (self));
      break;

    case PROP_TITLE:
      g_value_set_string (value, foundry_workspace_child_get_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_workspace_child_set_property (GObject      *object,
                                      guint         prop_id,
                                      const GValue *value,
                                      GParamSpec   *pspec)
{
  FoundryWorkspaceChild *self = FOUNDRY_WORKSPACE_CHILD (object);

  switch (prop_id)
    {
    case PROP_AREA:
      self->area = g_value_get_enum (value);
      break;

    case PROP_CHILD:
      foundry_workspace_child_set_child (self, g_value_get_object (value));
      break;

    case PROP_ICON:
      foundry_workspace_child_set_icon (self, g_value_get_object (value));
      break;

    case PROP_KIND:
      self->kind = g_value_get_enum (value);
      break;

    case PROP_MODIFIED:
      foundry_workspace_child_set_modified (self, g_value_get_boolean (value));
      break;

    case PROP_SUBTITLE:
      foundry_workspace_child_set_subtitle (self, g_value_get_string (value));
      break;

    case PROP_TITLE:
      foundry_workspace_child_set_title (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_workspace_child_class_init (FoundryWorkspaceChildClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = foundry_workspace_child_dispose;
  object_class->get_property = foundry_workspace_child_get_property;
  object_class->set_property = foundry_workspace_child_set_property;

  properties[PROP_AREA] =
    g_param_spec_enum ("area", NULL, NULL,
                       PANEL_TYPE_AREA,
                       PANEL_AREA_CENTER,
                       (G_PARAM_READWRITE |
                        G_PARAM_CONSTRUCT_ONLY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_CHILD] =
    g_param_spec_object ("child", NULL, NULL,
                         GTK_TYPE_WIDGET,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ICON] =
    g_param_spec_object ("icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_KIND] =
    g_param_spec_enum ("kind", NULL, NULL,
                       FOUNDRY_TYPE_WORKSPACE_CHILD_KIND,
                       FOUNDRY_WORKSPACE_CHILD_PAGE,
                       (G_PARAM_READWRITE |
                        G_PARAM_CONSTRUCT_ONLY |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_MODIFIED] =
    g_param_spec_boolean ("modified", NULL, NULL,
                          FALSE,
                          (G_PARAM_READWRITE |
                           G_PARAM_EXPLICIT_NOTIFY |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_SUBTITLE] =
    g_param_spec_string ("subtitle", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_workspace_child_init (FoundryWorkspaceChild *self)
{
  self->narrow_widget = ADW_BIN (adw_bin_new ());
  g_object_ref_sink (self->narrow_widget);

  self->wide_widget = PANEL_WIDGET (panel_widget_new ());
  g_object_ref_sink (self->wide_widget);

  self->bindings = g_binding_group_new ();
  g_binding_group_bind (self->bindings, "title",
                        self->wide_widget, "title",
                        G_BINDING_SYNC_CREATE);
  g_binding_group_bind (self->bindings, "icon",
                        self->wide_widget, "icon",
                        G_BINDING_SYNC_CREATE);
}

GtkWidget *
foundry_workspace_child_get_child (FoundryWorkspaceChild *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self), NULL);

  return self->child;
}

void
foundry_workspace_child_set_child (FoundryWorkspaceChild *self,
                                   GtkWidget             *child)
{
  g_autoptr(GMenuModel) model = NULL;

  g_return_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self));
  g_return_if_fail (!child || GTK_IS_WIDGET (child));

  if (child == self->child)
    return;

  if (child != NULL)
    g_object_ref_sink (child);

  if (FOUNDRY_IS_PAGE (child))
    model = foundry_page_dup_menu (FOUNDRY_PAGE (child));

  g_clear_object (&self->child);
  self->child = child;

  panel_widget_set_child (self->wide_widget, child);
  panel_widget_set_menu_model (self->wide_widget, model);

  g_binding_group_set_source (self->bindings, child);

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CHILD]);
}

GIcon *
foundry_workspace_child_get_icon (FoundryWorkspaceChild *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self), NULL);

  return self->icon;
}

void
foundry_workspace_child_set_icon (FoundryWorkspaceChild *self,
                                  GIcon                 *icon)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self));
  g_return_if_fail (!icon || G_IS_ICON (icon));

  if (g_set_object (&self->icon, icon))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_ICON]);
}

const char *
foundry_workspace_child_get_subtitle (FoundryWorkspaceChild *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self), NULL);

  return self->subtitle;
}

void
foundry_workspace_child_set_subtitle (FoundryWorkspaceChild *self,
                                      const char            *subtitle)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self));

  if (g_set_str (&self->subtitle, subtitle))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_SUBTITLE]);
}

const char *
foundry_workspace_child_get_title (FoundryWorkspaceChild *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self), NULL);

  return self->title;
}

void
foundry_workspace_child_set_title (FoundryWorkspaceChild *self,
                                   const char            *title)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self));

  if (g_set_str (&self->title, title))
    g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_TITLE]);
}

FoundryWorkspaceChildKind
foundry_workspace_child_get_kind (FoundryWorkspaceChild *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self), 0);

  return self->kind;
}

void
foundry_workspace_child_set_layout (FoundryWorkspaceChild  *self,
                                    FoundryWorkspaceLayout  layout)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self));
  g_return_if_fail (layout == FOUNDRY_WORKSPACE_LAYOUT_NARROW ||
                    layout == FOUNDRY_WORKSPACE_LAYOUT_WIDE);

  if (layout == FOUNDRY_WORKSPACE_LAYOUT_WIDE)
    {
      adw_bin_set_child (self->narrow_widget, NULL);
      panel_widget_set_child (self->wide_widget, self->child);
    }
  else
    {
      panel_widget_set_child (self->wide_widget, NULL);
      adw_bin_set_child (self->narrow_widget, self->child);
    }
}

G_DEFINE_ENUM_TYPE (FoundryWorkspaceChildKind, foundry_workspace_child_kind,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_WORKSPACE_CHILD_PAGE, "page"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_WORKSPACE_CHILD_PANEL, "panel"))

G_DEFINE_ENUM_TYPE (FoundryWorkspaceLayout, foundry_workspace_layout,
                    G_DEFINE_ENUM_VALUE (FOUNDRY_WORKSPACE_LAYOUT_NARROW, "narrow"),
                    G_DEFINE_ENUM_VALUE (FOUNDRY_WORKSPACE_LAYOUT_WIDE, "wide"))

GtkWidget *
foundry_workspace_child_get_wide_widget (FoundryWorkspaceChild *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self), NULL);

  return GTK_WIDGET (self->wide_widget);
}

GtkWidget *
foundry_workspace_child_get_narrow_widget (FoundryWorkspaceChild *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self), NULL);

  return GTK_WIDGET (self->narrow_widget);
}

gboolean
foundry_workspace_child_get_modified (FoundryWorkspaceChild *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self), FALSE);

  return self->modified;
}

void
foundry_workspace_child_set_modified (FoundryWorkspaceChild *self,
                                      gboolean               modified)
{
  g_return_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self));

  modified = !!modified;

  if (self->modified != modified)
    {
      self->modified = modified;
      panel_widget_set_modified (self->wide_widget, modified);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_MODIFIED]);
    }
}

PanelArea
foundry_workspace_child_get_area (FoundryWorkspaceChild *self)
{
  g_return_val_if_fail (FOUNDRY_IS_WORKSPACE_CHILD (self), 0);

  return self->area;
}
