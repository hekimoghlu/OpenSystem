/*
 * mks-inhibitor.c
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "mks-inhibitor-private.h"

struct _MksInhibitor
{
  GObject parent_instance;
  GdkToplevel *toplevel;
  guint shortcuts_inhibited : 1;
};

G_DEFINE_FINAL_TYPE (MksInhibitor, mks_inhibitor, G_TYPE_OBJECT)

static void
notify_cb (GdkToplevel *toplevel,
           GParamSpec  *pspec,
           gpointer     user_data)
{
  gboolean shortcuts_inhibited;

  g_object_get (toplevel,
                "shortcuts-inhibited", &shortcuts_inhibited,
                NULL);

  g_debug ("Toplevel %p shortcuts-inhibited: %s\n",
           toplevel,
           shortcuts_inhibited ? "YES" : "NO");
}

static void
inhibit_shortcuts (MksInhibitor *self,
                   GdkEvent     *event)
{
  guint count;

  g_assert (MKS_IS_INHIBITOR (self));
  g_assert (GDK_IS_EVENT (event));
  g_assert (GDK_IS_TOPLEVEL (self->toplevel));
  g_assert (self->shortcuts_inhibited == FALSE);

  count = GPOINTER_TO_UINT (g_object_get_data (G_OBJECT (self->toplevel), "MKS_INHIBITOR_COUNT"));
  count++;
  g_object_set_data (G_OBJECT (self->toplevel), "MKS_INHIBITOR_COUNT", GUINT_TO_POINTER (count));

  if (count == 1)
    {
      g_signal_connect (self->toplevel,
                        "notify::shortcuts-inhibited",
                        G_CALLBACK (notify_cb),
                        NULL);
      gdk_toplevel_inhibit_system_shortcuts (self->toplevel, event);
      notify_cb (self->toplevel, NULL, NULL);
    }

  self->shortcuts_inhibited = TRUE;
}

static void
uninhibit_shortcuts (MksInhibitor *self)
{
  guint count;

  g_assert (MKS_IS_INHIBITOR (self));
  g_assert (GDK_IS_TOPLEVEL (self->toplevel));
  g_assert (self->shortcuts_inhibited == TRUE);

  count = GPOINTER_TO_UINT (g_object_get_data (G_OBJECT (self->toplevel), "MKS_INHIBITOR_COUNT"));
  count--;
  g_object_set_data (G_OBJECT (self->toplevel), "MKS_INHIBITOR_COUNT", GUINT_TO_POINTER (count));

  if (count == 0)
    {
      g_signal_handlers_disconnect_by_func (self->toplevel,
                                            G_CALLBACK (notify_cb),
                                            NULL);
      gdk_toplevel_restore_system_shortcuts (self->toplevel);
    }

  self->shortcuts_inhibited = FALSE;
}

static void
mks_inhibitor_dispose (GObject *object)
{
  MksInhibitor *self = (MksInhibitor *)object;

  if (self->shortcuts_inhibited)
    uninhibit_shortcuts (self);

  G_OBJECT_CLASS (mks_inhibitor_parent_class)->dispose (object);
}

static void
mks_inhibitor_class_init (MksInhibitorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->dispose = mks_inhibitor_dispose;
}

static void
mks_inhibitor_init (MksInhibitor *self)
{
}

MksInhibitor *
mks_inhibitor_new (GtkWidget *widget,
                   GdkEvent  *event)
{
  MksInhibitor *self;
  GdkSurface *surface;
  GtkNative *native;

  g_return_val_if_fail (GTK_IS_WIDGET (widget), NULL);
  g_return_val_if_fail (GDK_IS_EVENT (event), NULL);

  self = g_object_new (MKS_TYPE_INHIBITOR, NULL);

  if ((native = gtk_widget_get_native (widget)) &&
      (surface = gtk_native_get_surface (native)) &&
      GDK_IS_TOPLEVEL (surface))
    {
      if (g_set_object (&self->toplevel, GDK_TOPLEVEL (surface)))
        inhibit_shortcuts (self, event);
    }

  return self;
}

void
mks_inhibitor_uninhibit (MksInhibitor *self)
{
  g_return_if_fail (MKS_IS_INHIBITOR (self));

  if (self->shortcuts_inhibited)
    uninhibit_shortcuts (self);
}
