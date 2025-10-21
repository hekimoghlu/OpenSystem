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

#include "govf-disk.h"

struct _GovfDisk
{
	GObject			  parent_instance;

	gchar			 *capacity;
	gchar			 *disk_id;
	gchar			 *file_ref;
	gchar			 *format;
};

G_DEFINE_TYPE (GovfDisk, govf_disk, G_TYPE_OBJECT)

/**
 * govf_disk_get_capacity:
 * @self: a #GovfDisk
 *
 * Returns the disk's capacity.
 *
 * Returns: (transfer none): the capacity
 */
const gchar *
govf_disk_get_capacity (GovfDisk *self)
{
	return self->capacity;
}

/**
 * govf_disk_set_capacity:
 * @self: a #GovfDisk
 * @capacity: capacity for the disk
 *
 * Sets a new capacity for the disk.
 */
void
govf_disk_set_capacity (GovfDisk    *self,
                        const gchar *capacity)
{
	g_free (self->capacity);
	self->capacity = g_strdup (capacity);
}

/**
 * govf_disk_get_disk_id:
 * @self: a #GovfDisk
 *
 * Returns the disk id.
 *
 * Returns: (transfer none): the disk id
 */
const gchar *
govf_disk_get_disk_id (GovfDisk *self)
{
	return self->disk_id;
}

/**
 * govf_disk_set_disk_id:
 * @self: a #GovfDisk
 * @disk_id: disk id for the disk
 *
 * Sets a new disk id for the disk.
 */
void
govf_disk_set_disk_id (GovfDisk    *self,
                       const gchar *disk_id)
{
	g_free (self->disk_id);
	self->disk_id = g_strdup (disk_id);
}

/**
 * govf_disk_get_file_ref:
 * @self: a #GovfDisk
 *
 * Returns the disk's file reference.
 *
 * Returns: (transfer none): the file ref
 */
const gchar *
govf_disk_get_file_ref (GovfDisk *self)
{
	return self->file_ref;
}

/**
 * govf_disk_set_file_ref:
 * @self: a #GovfDisk
 * @file_ref: file ref for the disk
 *
 * Sets a new file reference for the disk.
 */
void
govf_disk_set_file_ref (GovfDisk    *self,
                        const gchar *file_ref)
{
	g_free (self->file_ref);
	self->file_ref = g_strdup (file_ref);
}

/**
 * govf_disk_get_format:
 * @self: a #GovfDisk
 *
 * Returns the disk's format.
 *
 * Returns: (transfer none): the format
 */
const gchar *
govf_disk_get_format (GovfDisk *self)
{
	return self->format;
}

/**
 * govf_disk_set_format:
 * @self: a #GovfDisk
 * @format: format for the disk
 *
 * Sets a new format for the disk.
 */
void
govf_disk_set_format (GovfDisk    *self,
                      const gchar *format)
{
	g_free (self->format);
	self->format = g_strdup (format);
}

/**
 * govf_disk_new:
 *
 * Creates a new #GovfDisk.
 *
 * Returns: (transfer full): a #GovfDisk
 */
GovfDisk *
govf_disk_new (void)
{
	return g_object_new (GOVF_TYPE_DISK, NULL);
}

static void
govf_disk_finalize (GObject *object)
{
	GovfDisk *self = GOVF_DISK (object);

	g_free (self->capacity);
	g_free (self->disk_id);
	g_free (self->file_ref);
	g_free (self->format);

	G_OBJECT_CLASS (govf_disk_parent_class)->finalize (object);
}

static void
govf_disk_class_init (GovfDiskClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS (klass);

	object_class->finalize = govf_disk_finalize;
}

static void
govf_disk_init (GovfDisk *self)
{
}
