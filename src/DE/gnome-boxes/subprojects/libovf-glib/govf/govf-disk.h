/*
 * Copyright (C) 2016 Kalev Lember <klember@redhat.com>
 *
 * Licensed under the GNU Lesser General Public License Version 2.1
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 */

#ifndef __GOVF_DISK_H__
#define __GOVF_DISK_H__

#include <glib-object.h>

G_BEGIN_DECLS

#define GOVF_TYPE_DISK (govf_disk_get_type ())
G_DECLARE_FINAL_TYPE (GovfDisk, govf_disk, GOVF, DISK, GObject)

GovfDisk		 *govf_disk_new				(void);
const gchar		 *govf_disk_get_capacity		(GovfDisk	 *self);
void			  govf_disk_set_capacity		(GovfDisk	 *self,
								 const gchar	 *capacity);
const gchar		 *govf_disk_get_disk_id			(GovfDisk	 *self);
void			  govf_disk_set_disk_id			(GovfDisk	 *self,
								 const gchar	 *disk_id);
const gchar		 *govf_disk_get_file_ref		(GovfDisk	 *self);
void			  govf_disk_set_file_ref		(GovfDisk	 *self,
								 const gchar	 *file_ref);
const gchar		 *govf_disk_get_format			(GovfDisk	 *self);
void			  govf_disk_set_format			(GovfDisk	 *self,
								 const gchar	 *format);

G_END_DECLS

#endif /* __GOVF_DISK_H__ */

