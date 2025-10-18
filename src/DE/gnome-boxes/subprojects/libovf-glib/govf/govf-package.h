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

#ifndef __GOVF_PACKAGE_H__
#define __GOVF_PACKAGE_H__

#include "govf-disk.h"

#include <gio/gio.h>
#include <glib.h>
#include <glib-object.h>

G_BEGIN_DECLS

#define GOVF_TYPE_PACKAGE (govf_package_get_type ())
G_DECLARE_FINAL_TYPE (GovfPackage, govf_package, GOVF, PACKAGE, GObject)

#define GOVF_PACKAGE_ERROR (govf_package_error_quark ())

typedef enum
{
	GOVF_PACKAGE_ERROR_FAILED,
	GOVF_PACKAGE_ERROR_NOT_FOUND,
	GOVF_PACKAGE_ERROR_XML,
	GOVF_PACKAGE_ERROR_LAST
} GovfPackageError;

GQuark			  govf_package_error_quark		(void);

GovfPackage		 *govf_package_new			(void);
void    		  govf_package_load_from_ova_file	(GovfPackage		 *self,
								 const gchar		 *filename,
								 GCancellable		 *cancellable,
								 GAsyncReadyCallback	  callback,
								 gpointer		  user_data);
gboolean		  govf_package_load_from_ova_file_finish (GovfPackage		 *self,
								 GAsyncResult		 *result,
								 GError			**error);
gboolean		  govf_package_load_from_file		(GovfPackage		 *self,
								 const gchar		 *filename,
								 GError			**error);
gboolean		  govf_package_load_from_data		(GovfPackage		 *self,
								 const gchar		 *data,
								 gssize			  length,
								 GError			**error);
gboolean		  govf_package_save_file		(GovfPackage		 *self,
								 const gchar		 *filename,
								 GError			**error);
GPtrArray		 *govf_package_get_disks		(GovfPackage		 *self);
void    		  govf_package_extract_disk		(GovfPackage		 *self,
								 GovfDisk		 *disk,
								 const gchar		 *save_path,
								 GCancellable		 *cancellable,
								 GAsyncReadyCallback	  callback,
								 gpointer		  user_data);
gboolean		  govf_package_extract_disk_finish	(GovfPackage		 *self,
								 GAsyncResult		 *result,
								 GError			**error);

G_END_DECLS

#endif /* __GOVF_PACKAGE_H__ */
