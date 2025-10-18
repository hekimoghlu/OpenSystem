/*
 * Copyright (C) 2014 Carlos Garnacho <carlosg@gnome.org>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#include "tracker-extract-persistence.h"

struct _TrackerExtractPersistence
{
	GObject parent_instance;
	int fd;
};

G_DEFINE_TYPE (TrackerExtractPersistence, tracker_extract_persistence, G_TYPE_OBJECT)

static void
tracker_extract_persistence_finalize (GObject *object)
{
	TrackerExtractPersistence *persistence =
		TRACKER_EXTRACT_PERSISTENCE (object);

	if (persistence->fd >= 0)
		close (persistence->fd);

	G_OBJECT_CLASS (tracker_extract_persistence_parent_class)->finalize (object);
}

static void
tracker_extract_persistence_class_init (TrackerExtractPersistenceClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS (klass);

	object_class->finalize = tracker_extract_persistence_finalize;
}

static void
tracker_extract_persistence_init (TrackerExtractPersistence *persistence)
{
	persistence->fd = -1;
}

TrackerExtractPersistence *
tracker_extract_persistence_new (void)
{
	return g_object_new (TRACKER_TYPE_EXTRACT_PERSISTENCE,
	                     NULL);
}

void
tracker_extract_persistence_set_fd (TrackerExtractPersistence *persistence,
                                    int                        fd)
{
	if (persistence->fd >= 0)
		close (persistence->fd);
	persistence->fd = fd;
}

void
tracker_extract_persistence_set_file (TrackerExtractPersistence *persistence,
                                      GFile                     *file)
{
	g_autofree gchar *path = NULL;
	ssize_t retval;
	size_t len, written = 0;

	g_return_if_fail (TRACKER_IS_EXTRACT_PERSISTENCE (persistence));
	g_return_if_fail (!file || G_IS_FILE (file));

	if (file) {
		path = g_file_get_path (file);
	} else {
		path = g_strdup ("");
	}

	/* Write also the trailing \0 */
	len = strlen (path) + 1;

	lseek (persistence->fd, 0, SEEK_SET);

	while (TRUE) {
		retval = write (persistence->fd, &path[written], len - written);
		if (retval < 0 || written + retval >= len)
			break;

		written += retval;
	}
}

GFile *
tracker_extract_persistence_get_file (TrackerExtractPersistence *persistence)
{
	gchar buf[2048];
	int len;

	g_return_val_if_fail (TRACKER_IS_EXTRACT_PERSISTENCE (persistence), NULL);

	lseek (persistence->fd, 0, SEEK_SET);
	len = read (persistence->fd, buf, sizeof (buf));
	if (len <= 0)
		return NULL;
	if (buf[0] == '\0')
		return NULL;

	buf[len - 1] = '\0';
	return g_file_new_for_path (buf);
}
