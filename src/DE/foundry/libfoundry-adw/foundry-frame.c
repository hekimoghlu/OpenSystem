/* foundry-frame.c
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

#include "foundry-frame-private.h"
#include "foundry-page-private.h"
#include "foundry-workspace-private.h"

struct _FoundryFrame
{
  PanelFrame parent_instance;
  FoundryWorkspaceChildKind kind;
};

G_DEFINE_FINAL_TYPE (FoundryFrame, foundry_frame, PANEL_TYPE_FRAME)

static void
foundry_frame_notify_visible_child_cb (FoundryFrame *self,
                                       GParamSpec   *pspec)
{
  FoundryActionMuxer *page_actions = NULL;
  PanelWidget *visible_child;

  g_assert (FOUNDRY_IS_FRAME (self));

  if ((visible_child = panel_frame_get_visible_child (PANEL_FRAME (self))))
    {
      GtkWidget *child;

      g_assert (PANEL_IS_WIDGET (visible_child));

      child = panel_widget_get_child (visible_child);

      if (FOUNDRY_IS_PAGE (child))
        page_actions = _foundry_page_get_action_muxer (FOUNDRY_PAGE (child));
    }

  gtk_widget_insert_action_group (GTK_WIDGET (self),
                                  "current-page",
                                  G_ACTION_GROUP (page_actions));
}

static void
foundry_frame_page_closed (PanelFrame  *frame,
                           PanelWidget *page)
{
  FoundryFrame *self = (FoundryFrame *)frame;
  GtkWidget *ancestor;

  g_assert (FOUNDRY_IS_FRAME (self));
  g_assert (PANEL_IS_WIDGET (page));

  if ((ancestor = gtk_widget_get_ancestor (GTK_WIDGET (frame), FOUNDRY_TYPE_WORKSPACE)))
    {
      FoundryWorkspace *workspace = FOUNDRY_WORKSPACE (ancestor);

      _foundry_workspace_frame_page_closed (workspace, self, page, self->kind);
    }
}

static void
foundry_frame_class_init (FoundryFrameClass *klass)
{
  PanelFrameClass *frame_class = PANEL_FRAME_CLASS (klass);

  frame_class->page_closed = foundry_frame_page_closed;
}

static void
foundry_frame_init (FoundryFrame *self)
{
  g_signal_connect (self,
                    "notify::visible-child",
                    G_CALLBACK (foundry_frame_notify_visible_child_cb),
                    NULL);
}

GtkWidget *
foundry_frame_new (FoundryWorkspaceChildKind child_kind)
{
  FoundryFrame *self;

  self = g_object_new (FOUNDRY_TYPE_FRAME, NULL);
  self->kind = child_kind;

  if (child_kind == FOUNDRY_WORKSPACE_CHILD_PAGE)
    {
      GtkWidget *header = panel_frame_tab_bar_new ();
      panel_frame_set_header (PANEL_FRAME (self), PANEL_FRAME_HEADER (header));
    }

  return GTK_WIDGET (self);
}
