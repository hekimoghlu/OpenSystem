/*
 * Copyright (C) 2017-2020 Red Hat
 * Copyright (C) 2020 Alberts Muktupāvels
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef GF_CRTC_MODE_PRIVATE_H
#define GF_CRTC_MODE_PRIVATE_H

#include <glib-object.h>
#include <stdint.h>

#include "gf-crtc-mode-info-private.h"

G_BEGIN_DECLS

#define GF_TYPE_CRTC_MODE (gf_crtc_mode_get_type ())
G_DECLARE_DERIVABLE_TYPE (GfCrtcMode, gf_crtc_mode, GF, CRTC_MODE, GObject)

struct _GfCrtcModeClass
{
  GObjectClass parent_class;
};

uint64_t              gf_crtc_mode_get_id   (GfCrtcMode *self);

const char           *gf_crtc_mode_get_name (GfCrtcMode *self);

const GfCrtcModeInfo *gf_crtc_mode_get_info (GfCrtcMode *self);

G_END_DECLS

#endif
