/*
 * mks-cairo-framebuffer-private.h
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

#include <cairo.h>
#include <gdk/gdk.h>

G_BEGIN_DECLS

#define MKS_TYPE_CAIRO_FRAMEBUFFER (mks_cairo_framebuffer_get_type())

G_DECLARE_FINAL_TYPE (MksCairoFramebuffer, mks_cairo_framebuffer, MKS, CAIRO_FRAMEBUFFER, GObject)

MksCairoFramebuffer *mks_cairo_framebuffer_new        (cairo_format_t       format,
                                                       guint                width,
                                                       guint                height);
cairo_format_t       mks_cairo_framebuffer_get_format (MksCairoFramebuffer *self);
guint                mks_cairo_framebuffer_get_width  (MksCairoFramebuffer *self);
guint                mks_cairo_framebuffer_get_height (MksCairoFramebuffer *self);
cairo_t             *mks_cairo_framebuffer_update     (MksCairoFramebuffer *self,
                                                       guint                x,
                                                       guint                y,
                                                       guint                width,
                                                       guint                height);
void                 mks_cairo_framebuffer_copy_to    (MksCairoFramebuffer *self,
                                                       MksCairoFramebuffer *dest);
void                 mks_cairo_framebuffer_clear      (MksCairoFramebuffer *self);

G_END_DECLS
