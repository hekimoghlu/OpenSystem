/*
 * mks-display.h
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

#pragma once

#include <gtk/gtk.h>

#include "mks-types.h"
#include "mks-version-macros.h"

G_BEGIN_DECLS

#define MKS_TYPE_DISPLAY (mks_display_get_type())

MKS_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (MksDisplay, mks_display, MKS, DISPLAY, GtkWidget)

struct _MksDisplayClass
{
  GtkWidgetClass parent_class;

  /*< private >*/
  gpointer _reserved[16];
};

MKS_AVAILABLE_IN_ALL
GtkWidget          *mks_display_new                (void);
MKS_AVAILABLE_IN_ALL
GtkShortcutTrigger *mks_display_get_ungrab_trigger (MksDisplay         *self);
MKS_AVAILABLE_IN_ALL
void                mks_display_set_ungrab_trigger (MksDisplay         *self,
                                                    GtkShortcutTrigger *trigger);
MKS_AVAILABLE_IN_ALL
MksScreen          *mks_display_get_screen         (MksDisplay         *self);
MKS_AVAILABLE_IN_ALL
void                mks_display_set_screen         (MksDisplay         *self,
                                                    MksScreen          *screen);
MKS_AVAILABLE_IN_ALL
gboolean            mks_display_get_auto_resize    (MksDisplay         *self);
MKS_AVAILABLE_IN_ALL
void                mks_display_set_auto_resize    (MksDisplay         *self,
                                                    gboolean            auto_resize);
MKS_AVAILABLE_IN_ALL
gboolean
          mks_display_get_event_position_in_guest  (MksDisplay         *self,
                                                    GdkEvent           *event,
                                                    double             *guest_x,
                                                    double             *guest_y);
G_END_DECLS
