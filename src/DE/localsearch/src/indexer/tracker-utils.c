/*
 * Copyright (C) 2009, Nokia <ivan.frade@nokia.com>
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
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#include "config-miners.h"

#include "tracker-utils.h"

#define QUERY_RESOURCE "/org/freedesktop/Tracker3/Miner/Files/queries/"

TrackerSparqlStatement *
tracker_load_statement (TrackerSparqlConnection  *conn,
                        const gchar              *query_filename,
                        GError                  **error)
{
	g_autofree gchar *resource_path = NULL;

	resource_path = g_strconcat (QUERY_RESOURCE, query_filename, NULL);

	return tracker_sparql_connection_load_statement_from_gresource (conn,
	                                                                resource_path,
	                                                                NULL,
	                                                                error);
}
