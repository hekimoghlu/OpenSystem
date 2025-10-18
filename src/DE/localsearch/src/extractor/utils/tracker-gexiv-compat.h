/*
 * Copyright (C) 2025 Red Hat, Inc.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 *
 * Author: Carlos Garnacho <carlosg@gnome.org>
 */
#pragma once

#include "config-miners.h"

#include <gexiv2/gexiv2.h>

/* Compatibility defines for older versions of gexiv2 */

#if !GEXIV2_CHECK_VERSION (0, 12, 2)
#define gexiv2_metadata_get_tag_string(m,s,e) (gexiv2_metadata_get_tag_string ((m), (s)))
#define gexiv2_metadata_get_exposure_time(m,n,d,e) (gexiv2_metadata_get_exposure_time ((m), (n), (d)))
#define gexiv2_metadata_get_fnumber(m,e) (gexiv2_metadata_get_fnumber ((m)))
#define gexiv2_metadata_has_tag(m,t,e) (gexiv2_metadata_has_tag ((m), (t)))
#define gexiv2_metadata_get_tag_long(m,t,e) (gexiv2_metadata_get_tag_long ((m), (t)))
#define gexiv2_metadata_get_focal_length(m,e) (gexiv2_metadata_get_focal_length ((m)))
#define gexiv2_metadata_get_iso_speed(m,e) (gexiv2_metadata_get_iso_speed ((m)))
#define gexiv2_metadata_get_orientation(m,e) (gexiv2_metadata_get_orientation ((m)))
#define gexiv2_metadata_get_gps_altitude(m,e) (gexiv2_metadata_get_gps_altitude ((m), &tmp_gps) ? tmp_gps : NAN)
#define gexiv2_metadata_get_gps_latitude(m,e) (gexiv2_metadata_get_gps_latitude ((m), &tmp_gps) ? tmp_gps : NAN)
#define gexiv2_metadata_get_gps_longitude(m,e) (gexiv2_metadata_get_gps_longitude ((m), &tmp_gps) ? tmp_gps : NAN)
#elif !GEXIV2_CHECK_VERSION (0, 16, 0)
#define gexiv2_metadata_get_tag_string(m,s,e) (gexiv2_metadata_try_get_tag_string ((m), (s), (e)))
#define gexiv2_metadata_get_exposure_time(m,n,d,e) (gexiv2_metadata_try_get_exposure_time ((m), (n), (d), (e)))
#define gexiv2_metadata_get_fnumber(m,e) (gexiv2_metadata_try_get_fnumber ((m), (e)))
#define gexiv2_metadata_has_tag(m,t,e) (gexiv2_metadata_try_has_tag ((m), (t), (e)))
#define gexiv2_metadata_get_tag_long(m,t,e) (gexiv2_metadata_try_get_tag_long ((m), (t), (e)))
#define gexiv2_metadata_get_focal_length(m,e) (gexiv2_metadata_try_get_focal_length ((m), (e)))
#define gexiv2_metadata_get_iso_speed(m,e) (gexiv2_metadata_try_get_iso_speed ((m), (e)))
#define gexiv2_metadata_get_orientation(m,e) (gexiv2_metadata_try_get_orientation ((m), (e)))
#define gexiv2_metadata_get_gps_altitude(m,e) (gexiv2_metadata_try_get_gps_altitude ((m), &tmp_gps, (e)) ? tmp_gps : NAN)
#define gexiv2_metadata_get_gps_latitude(m,e) (gexiv2_metadata_try_get_gps_latitude ((m), &tmp_gps, (e)) ? tmp_gps : NAN)
#define gexiv2_metadata_get_gps_longitude(m,e) (gexiv2_metadata_try_get_gps_longitude ((m), &tmp_gps, (e)) ? tmp_gps : NAN)
#endif
