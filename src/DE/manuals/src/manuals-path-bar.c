/*
 * manuals-path-bar.c
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

#include "manuals-path-bar.h"

#include "manuals-path-button.h"
#include "manuals-path-element.h"
#include "manuals-path-model.h"

struct _ManualsPathBar
{
  GtkWidget             parent_instance;

  FoundryDocumentation *navigatable;
  ManualsPathModel     *model;

  GtkBox               *elements;
  GtkScrolledWindow    *scroller;

  int                   inhibit_scroll;
  guint                 scroll_source;
};

enum {
  PROP_0,
  PROP_NAVIGATABLE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (ManualsPathBar, manuals_path_bar, GTK_TYPE_WIDGET)

static GParamSpec *properties[N_PROPS];

static void
manuals_path_bar_scroll_to_end (ManualsPathBar *self)
{
  GtkAdjustment *hadj;
  double page_size;
  double upper;

  g_assert (MANUALS_IS_PATH_BAR (self));

  if (self->inhibit_scroll)
    return;

  hadj = gtk_scrolled_window_get_hadjustment (self->scroller);
  upper = gtk_adjustment_get_upper (hadj);
  page_size = gtk_adjustment_get_page_size (hadj);

  gtk_adjustment_set_value (hadj, upper - page_size);
}

static gboolean
manuals_path_bar_scroll_to_end_idle (gpointer data)
{
  ManualsPathBar *self = data;

  g_assert (MANUALS_IS_PATH_BAR (self));

  self->scroll_source = 0;
  manuals_path_bar_scroll_to_end (self);
  return G_SOURCE_REMOVE;
}

static void
manuals_path_bar_queue_scroll (ManualsPathBar *self)
{
  g_assert (MANUALS_IS_PATH_BAR (self));

  g_clear_handle_id (&self->scroll_source, g_source_remove);
  self->scroll_source = g_idle_add_full (G_PRIORITY_LOW,
                                         manuals_path_bar_scroll_to_end_idle,
                                         g_object_ref (self),
                                         g_object_unref);
}

static void
manuals_path_bar_notify_upper_cb (ManualsPathBar *self,
                                  GParamSpec            *pspec,
                                  GtkAdjustment         *hadj)
{
  GtkWidget *focus;
  GtkRoot *root;

  g_assert (MANUALS_IS_PATH_BAR (self));
  g_assert (GTK_IS_ADJUSTMENT (hadj));

  root = gtk_widget_get_root (GTK_WIDGET (self));
  focus = gtk_root_get_focus (root);

  if (focus && gtk_widget_is_ancestor (focus, GTK_WIDGET (self)))
    return;

  manuals_path_bar_queue_scroll (self);
}

static GtkWidget *
create_button (ManualsPathElement *element)
{
  g_autoptr(ManualsPathElement) to_free = element;

  g_assert (MANUALS_IS_PATH_ELEMENT (element));

  return g_object_new (MANUALS_TYPE_PATH_BUTTON,
                       "element", element,
                       "valign", GTK_ALIGN_CENTER,
                       NULL);
}

static void
manuals_path_bar_path_items_changed_cb (ManualsPathBar *self,
                                        guint                  position,
                                        guint                  removed,
                                        guint                  added,
                                        ManualsPathModel      *model)
{
  g_assert (MANUALS_IS_PATH_BAR (self));
  g_assert (MANUALS_IS_PATH_MODEL (model));

  if (removed > 0)
    {
      GtkWidget *child = gtk_widget_get_first_child (GTK_WIDGET (self->elements));

      for (guint j = position; j > 0; j--)
        child = gtk_widget_get_next_sibling (child);

      while (removed > 0)
        {
          GtkWidget *to_remove = child;
          child = gtk_widget_get_next_sibling (child);
          gtk_widget_unparent (to_remove);
          removed--;
        }
    }

  if (added > 0)
    {
      GtkWidget *child = gtk_widget_get_first_child (GTK_WIDGET (self->elements));

      for (guint j = position; j > 0; j--)
        child = gtk_widget_get_next_sibling (child);

      for (guint i = 0; i < added; i++)
        {
          GtkWidget *to_add = create_button (g_list_model_get_item (G_LIST_MODEL (model), position + i));
          gtk_box_insert_child_after (self->elements, to_add, child);
          child = to_add;
        }
    }
}

static void
manuals_path_bar_dispose (GObject *object)
{
  ManualsPathBar *self = (ManualsPathBar *)object;
  GtkWidget *child;

  gtk_widget_dispose_template (GTK_WIDGET (self), MANUALS_TYPE_PATH_BAR);

  while ((child = gtk_widget_get_first_child (GTK_WIDGET (self))))
    gtk_widget_unparent (child);

  g_clear_object (&self->navigatable);
  g_clear_object (&self->model);

  G_OBJECT_CLASS (manuals_path_bar_parent_class)->dispose (object);
}

static void
manuals_path_bar_get_property (GObject    *object,
                                  guint       prop_id,
                                  GValue     *value,
                                  GParamSpec *pspec)
{
  ManualsPathBar *self = MANUALS_PATH_BAR (object);

  switch (prop_id)
    {
    case PROP_NAVIGATABLE:
      g_value_set_object (value, manuals_path_bar_get_navigatable (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
manuals_path_bar_set_property (GObject      *object,
                                  guint         prop_id,
                                  const GValue *value,
                                  GParamSpec   *pspec)
{
  ManualsPathBar *self = MANUALS_PATH_BAR (object);

  switch (prop_id)
    {
    case PROP_NAVIGATABLE:
      manuals_path_bar_set_navigatable (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
manuals_path_bar_class_init (ManualsPathBarClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  object_class->dispose = manuals_path_bar_dispose;
  object_class->get_property = manuals_path_bar_get_property;
  object_class->set_property = manuals_path_bar_set_property;

  properties[PROP_NAVIGATABLE] =
    g_param_spec_object ("navigatable", NULL, NULL,
                         FOUNDRY_TYPE_DOCUMENTATION,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  gtk_widget_class_set_layout_manager_type (widget_class, GTK_TYPE_BIN_LAYOUT);
  gtk_widget_class_set_css_name (widget_class, "ManualsPathBar");
  gtk_widget_class_set_template_from_resource (widget_class, "/app/devsuite/manuals/manuals-path-bar.ui");
  gtk_widget_class_bind_template_child (widget_class, ManualsPathBar, elements);
  gtk_widget_class_bind_template_child (widget_class, ManualsPathBar, scroller);
}

static void
manuals_path_bar_init (ManualsPathBar *self)
{
  guint n_items;

  self->model = manuals_path_model_new ();

  g_signal_connect_object (self->model,
                           "items-changed",
                           G_CALLBACK (manuals_path_bar_path_items_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  gtk_widget_init_template (GTK_WIDGET (self));

  n_items = g_list_model_get_n_items (G_LIST_MODEL (self->model));

  if (n_items > 0)
    manuals_path_bar_path_items_changed_cb (self, 0, n_items, 0, self->model);

  g_signal_connect_object (gtk_scrolled_window_get_hadjustment (self->scroller),
                           "notify::upper",
                           G_CALLBACK (manuals_path_bar_notify_upper_cb),
                           self,
                           G_CONNECT_SWAPPED);
}

ManualsPathBar *
manuals_path_bar_new (void)
{
  return g_object_new (MANUALS_TYPE_PATH_BAR, NULL);
}

FoundryDocumentation *
manuals_path_bar_get_navigatable (ManualsPathBar *self)
{
  g_return_val_if_fail (MANUALS_IS_PATH_BAR (self), NULL);

  return self->navigatable;
}

void
manuals_path_bar_set_navigatable (ManualsPathBar       *self,
                                  FoundryDocumentation *navigatable)
{
  g_return_if_fail (MANUALS_IS_PATH_BAR (self));

  if (g_set_object (&self->navigatable, navigatable))
    {
      manuals_path_model_set_navigatable (self->model, navigatable);
      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_NAVIGATABLE]);
    }
}

void
manuals_path_bar_inhibit_scroll (ManualsPathBar *self)
{
  g_return_if_fail (MANUALS_IS_PATH_BAR (self));

  self->inhibit_scroll++;
}

void
manuals_path_bar_uninhibit_scroll (ManualsPathBar *self)
{
  g_return_if_fail (MANUALS_IS_PATH_BAR (self));

  self->inhibit_scroll--;
}

