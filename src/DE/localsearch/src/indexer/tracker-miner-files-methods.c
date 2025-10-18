/*
 * Copyright (C) 2008, Nokia <ivan.frade@nokia.com>
 * Copyright (C) 2021, Red Hat Inc.
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
 */

#include "config-miners.h"

#include <gio/gunixmounts.h>

#include "tracker-miner-files-methods.h"

#include <tracker-common.h>

#define DEFAULT_GRAPH "tracker:FileSystem"

#define DIRECTORY_MIME "inode/directory"

static void
miner_files_add_to_datasource (TrackerMinerFiles *mf,
                               GFile             *file,
                               TrackerResource   *resource)
{
	TrackerIndexingTree *indexing_tree;
	TrackerMinerFS *fs = TRACKER_MINER_FS (mf);
	const char *datasource_uri = NULL;

	indexing_tree = tracker_miner_fs_get_indexing_tree (fs);

	if (tracker_indexing_tree_file_is_root (indexing_tree, file)) {
		datasource_uri = tracker_miner_fs_get_identifier (fs, file);
	} else {
		GFile *root;

		root = tracker_indexing_tree_get_root (indexing_tree, file, NULL, NULL);

		if (root)
			datasource_uri = tracker_miner_fs_get_identifier (fs, root);
	}

	if (datasource_uri)
		tracker_resource_set_uri (resource, "nie:dataSource", datasource_uri);
}

static void
miner_files_add_mount_info (TrackerMinerFiles *miner,
                            TrackerResource   *resource,
                            GFile             *file)
{
	TrackerIndexingTree *indexing_tree;
	TrackerDirectoryFlags flags;
	gboolean is_removable;

	indexing_tree = tracker_miner_fs_get_indexing_tree (TRACKER_MINER_FS (miner));
	tracker_indexing_tree_get_root (indexing_tree, file, NULL, &flags);
	is_removable = !!(flags & TRACKER_DIRECTORY_FLAG_IS_VOLUME);

	if (!is_removable)
		return;

	tracker_resource_set_boolean (resource, "tracker:isRemovable", is_removable);
}

static void
maybe_add_root_info (TrackerMinerFS  *fs,
                     GFile           *file,
                     TrackerResource *resource)
{
	TrackerIndexingTree *indexing_tree =
		tracker_miner_fs_get_indexing_tree (fs);

	if (!tracker_indexing_tree_file_is_root (indexing_tree, file))
		return;

	tracker_resource_set_uri (resource, "rdf:type", "tracker:IndexedFolder");
	tracker_resource_set_uri (resource, "nie:rootElementOf",
	                          tracker_resource_get_identifier (resource));
	tracker_resource_set_boolean (resource, "tracker:available", TRUE);
}

static TrackerResource *
miner_files_create_folder_information_element (TrackerMinerFiles *miner,
                                               GFile             *file)
{
	TrackerMinerFS *fs = TRACKER_MINER_FS (miner);
	TrackerResource *resource;
	TrackerIndexingTree *indexing_tree;
	const gchar *urn;
	g_autofree char *uri = NULL;

	/* Preserve URN for nfo:Folders */
	urn = tracker_miner_fs_get_identifier (fs, file);
	resource = tracker_resource_new (urn);

	tracker_resource_set_string (resource, "nie:mimeType", DIRECTORY_MIME);
	tracker_resource_add_uri (resource, "rdf:type", "nie:InformationElement");
	tracker_resource_add_uri (resource, "rdf:type", "nfo:Folder");

	uri = g_file_get_uri (file);
	tracker_resource_set_uri (resource, "nie:isStoredAs", uri);

	indexing_tree = tracker_miner_fs_get_indexing_tree (TRACKER_MINER_FS (miner));

	if (tracker_indexing_tree_file_is_root (indexing_tree, file)) {
		maybe_add_root_info (fs, file, resource);
		miner_files_add_mount_info (miner, resource, file);
	}

	return resource;
}

static TrackerResource *
miner_files_create_text_file_information_element (TrackerMinerFiles *miner,
                                                  GFile             *file,
                                                  const gchar       *mime_type)
{
	TrackerResource *resource;
	GStrv rdf_types;
	const gchar *urn;
	int i;

	urn = tracker_miner_fs_get_identifier (TRACKER_MINER_FS (miner),
	                                       file);
	resource = tracker_resource_new (urn);

	rdf_types = tracker_extract_module_manager_get_rdf_types (mime_type);

	for (i = 0; rdf_types[i]; i++)
		tracker_resource_add_uri (resource, "rdf:type", rdf_types[i]);

	g_strfreev (rdf_types);

	return resource;
}

static TrackerResource *
miner_files_create_empty_information_element (TrackerMinerFiles *miner,
                                              GFile             *file,
                                              const gchar       *mime_type)
{
	TrackerResource *resource;
	const gchar *urn;

	urn = tracker_miner_fs_get_identifier (TRACKER_MINER_FS (miner),
	                                       file);
	resource = tracker_resource_new (urn);
	tracker_resource_add_uri (resource, "rdf:type", "nie:InformationElement");
	tracker_resource_set_string (resource, "nie:mimeType", mime_type);

	return resource;
}

static gchar *
get_content_type (GFile *file)
{
	g_autoptr (GFileInfo) info = NULL;

	info = g_file_query_info (file,
	                          G_FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE,
	                          G_FILE_QUERY_INFO_NOFOLLOW_SYMLINKS,
	                          NULL, NULL);

	if (!info || !g_file_info_has_attribute (info, G_FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE))
		return NULL;

	return g_strdup (g_file_info_get_content_type (info));
}

void
tracker_miner_files_process_file (TrackerMinerFS      *fs,
                                  GFile               *file,
                                  GFileInfo           *file_info,
                                  TrackerSparqlBuffer *buffer,
                                  gboolean             create)
{
	g_autoptr (TrackerResource) resource = NULL, graph_file = NULL;
	const gchar *graph = NULL;
	const gchar *parent_urn;
	g_autoptr (GFile) parent = NULL;
	g_autofree gchar *uri = NULL;
	g_autofree gchar *mime_type = NULL;
	g_autoptr (GDateTime) modified = NULL;
	g_autoptr (GDateTime) accessed = NULL, created = NULL;

	mime_type = get_content_type (file);

	uri = g_file_get_uri (file);

	modified = g_file_info_get_modification_date_time (file_info);
	if (!modified)
		modified = g_date_time_new_from_unix_utc (0);

	resource = tracker_resource_new (uri);

	tracker_resource_add_uri (resource, "rdf:type", "nfo:FileDataObject");

	parent = g_file_get_parent (file);
	parent_urn = tracker_miner_fs_get_identifier (fs, parent);

	if (parent_urn) {
		tracker_resource_set_uri (resource, "nfo:belongsToContainer", parent_urn);
	}

	tracker_resource_set_string (resource, "nfo:fileName",
	                             g_file_info_get_display_name (file_info));
	tracker_resource_set_int64 (resource, "nfo:fileSize",
	                            g_file_info_get_size (file_info));

	tracker_resource_set_datetime (resource, "nfo:fileLastModified", modified);

#ifdef GIO_SUPPORTS_CREATION_TIME
	accessed = g_file_info_get_access_date_time (file_info);
	if (accessed)
		tracker_resource_set_datetime (resource, "nfo:fileLastAccessed", accessed);

	created = g_file_info_get_creation_date_time (file_info);
	if (created)
		tracker_resource_set_datetime (resource, "nfo:fileCreated", created);
#else
	time_ = (time_t) g_file_info_get_attribute_uint64 (file_info, G_FILE_ATTRIBUTE_TIME_ACCESS);
	accessed = g_date_time_new_from_unix_local (time_);
	if (accessed)
		tracker_resource_set_datetime (resource, "nfo:fileLastAccessed", accessed);
#endif

	/* The URL of the DataObject (because IE = DO, this is correct) */
	tracker_resource_set_string (resource, "nie:url", uri);

	miner_files_add_to_datasource (TRACKER_MINER_FILES (fs), file, resource);

	if (mime_type)
		graph = tracker_extract_module_manager_get_graph (mime_type);

	if (mime_type && graph) {
		TrackerIndexingTree *indexing_tree;
		TrackerResource *information_element;

		/* This mimetype will be extracted by some module, pre-fill the
		 * nfo:FileDataObject in that graph.
		 * Empty files skipped as mime-type for those cannot be trusted.
		 */
		graph_file = tracker_resource_new (uri);
		tracker_resource_add_uri (graph_file, "rdf:type", "nfo:FileDataObject");

		tracker_resource_set_string (graph_file, "nfo:fileName",
		                             g_file_info_get_display_name (file_info));

		tracker_resource_set_datetime (graph_file, "nfo:fileLastModified", modified);

		tracker_resource_set_int64 (graph_file, "nfo:fileSize",
		                            g_file_info_get_size (file_info));
		miner_files_add_to_datasource (TRACKER_MINER_FILES (fs), file, graph_file);

		indexing_tree = tracker_miner_fs_get_indexing_tree (fs);

		if (tracker_extract_module_manager_check_fallback_rdf_type (mime_type,
		                                                            "nfo:PlainTextDocument") &&
		    !tracker_indexing_tree_file_has_allowed_text_extension (indexing_tree, file)) {
			/* We let disallowed text files have a shallow document nie:InformationElement */
			information_element =
				miner_files_create_text_file_information_element (TRACKER_MINER_FILES (fs),
				                                                  file, mime_type);

			tracker_resource_set_string (resource, "tracker:extractorHash",
			                             tracker_extract_module_manager_get_hash (mime_type));
		} else {
			/* Insert only the base nie:InformationElement class, for the extractor to get
			 * the suitable content identifier.
			 */
			information_element =
				miner_files_create_empty_information_element (TRACKER_MINER_FILES (fs),
				                                              file,
				                                              mime_type);
		}

		tracker_resource_set_take_relation (graph_file, "nie:interpretedAs", information_element);
		tracker_resource_set_uri (information_element, "nie:isStoredAs", uri);
	} else if (g_file_info_get_file_type (file_info) == G_FILE_TYPE_DIRECTORY) {
		g_autoptr (TrackerResource) folder = NULL;
		const char *urn = NULL;

		urn = tracker_miner_fs_get_identifier (fs, file);

		folder = tracker_resource_new (urn);
		tracker_resource_set_uri (folder, "rdf:type", "nfo:Folder");
		maybe_add_root_info (fs, file, folder);

		tracker_resource_set_relation (resource, "nie:interpretedAs", folder);
		tracker_resource_set_uri (folder, "nie:isStoredAs", uri);
	}

	if (!create) {
		tracker_sparql_buffer_log_clear_content (buffer, file);
	}

	tracker_sparql_buffer_log_file (buffer, file,
	                                graph,
	                                resource,
	                                graph_file);
}

void
tracker_miner_files_process_file_attributes (TrackerMinerFS      *fs,
                                             GFile               *file,
                                             GFileInfo           *info,
                                             TrackerSparqlBuffer *buffer)
{
	g_autoptr (TrackerResource) resource = NULL, graph_file = NULL;
	g_autofree gchar *uri = NULL;
	g_autofree gchar *mime_type = NULL;
	const gchar *graph = NULL;
	g_autoptr (GDateTime) modified = NULL;
	g_autoptr (GDateTime) accessed = NULL, created = NULL;

	mime_type = get_content_type (file);

	uri = g_file_get_uri (file);
	resource = tracker_resource_new (uri);
	tracker_resource_add_uri (resource, "rdf:type", "nfo:FileDataObject");

	modified = g_file_info_get_modification_date_time (info);
	if (!modified)
		modified = g_date_time_new_from_unix_utc (0);

	if (mime_type)
		graph = tracker_extract_module_manager_get_graph (mime_type);

	/* Update nfo:fileLastModified */
	tracker_resource_set_datetime (resource, "nfo:fileLastModified", modified);
	if (graph) {
		graph_file = tracker_resource_new (uri);
		tracker_resource_add_uri (graph_file, "rdf:type", "nfo:FileDataObject");
		tracker_resource_set_datetime (graph_file, "nfo:fileLastModified", modified);
	}

#ifdef GIO_SUPPORTS_CREATION_TIME
	/* Update nfo:fileLastAccessed */
	accessed = g_file_info_get_access_date_time (info);
	tracker_resource_set_datetime (resource, "nfo:fileLastAccessed", accessed);

	/* Update nfo:fileCreated */
	created = g_file_info_get_creation_date_time (info);

	if (created)
		tracker_resource_set_datetime (resource, "nfo:fileCreated", created);
#else
	time_ = (time_t) g_file_info_get_attribute_uint64 (info, G_FILE_ATTRIBUTE_TIME_ACCESS);
	accessed = g_date_time_new_from_unix_local (time_);
	if (accessed)
		tracker_resource_set_datetime (resource, "nfo:fileLastAccessed", accessed);
#endif

	tracker_sparql_buffer_log_attributes_update (buffer,
	                                             file,
	                                             graph,
	                                             resource,
	                                             graph_file);
}

void
tracker_miner_files_finish_directory (TrackerMinerFS      *fs,
                                      GFile               *file,
                                      TrackerSparqlBuffer *buffer)
{
	TrackerIndexingTree *indexing_tree;
	g_autoptr (TrackerResource) resource = NULL, folder_resource = NULL;
	g_autofree char *uri = NULL;
	gboolean is_root;

	indexing_tree = tracker_miner_fs_get_indexing_tree (fs);
	is_root = tracker_indexing_tree_file_is_root (indexing_tree, file);

	uri = g_file_get_uri (file);
	resource = tracker_resource_new (uri);
	tracker_resource_set_string (resource, "tracker:extractorHash",
	                             tracker_extract_module_manager_get_hash (DIRECTORY_MIME));

	folder_resource =
		miner_files_create_folder_information_element (TRACKER_MINER_FILES (fs),
		                                               file);

	tracker_sparql_buffer_log_folder (buffer, file,
	                                  is_root,
	                                  resource, folder_resource);
}

gchar *
tracker_miner_files_get_content_identifier (TrackerMinerFiles *mf,
                                            GFile             *file,
                                            GFileInfo         *info)
{
	TrackerIndexingTree *indexing_tree;
	g_autofree gchar *inode = NULL, *str = NULL, *id = NULL;
	const char *root_id;

	indexing_tree = tracker_miner_fs_get_indexing_tree (TRACKER_MINER_FS (mf));
	tracker_indexing_tree_get_root (indexing_tree, file, &root_id, NULL);
	id = g_strdup (root_id);

	if (!id) {
		id = g_strdup (g_file_info_get_attribute_string (info,
		                                                 G_FILE_ATTRIBUTE_ID_FILESYSTEM));
	}

	inode = g_file_info_get_attribute_as_string (info, G_FILE_ATTRIBUTE_UNIX_INODE);

	/* Format:
	 * 'urn:fileid:' [uuid] ':' [inode]
	 */
	str = g_strconcat ("urn:fileid:", id, ":", inode, NULL);

	return g_steal_pointer (&str);
}
