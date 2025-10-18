/*
 * Copyright (C) 2025, Red Hat Inc.
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

#include "tracker-controller.h"

#include "tracker-config.h"

#include <tracker-common.h>

struct _TrackerController
{
	GObject parent_instance;

	TrackerIndexingTree *indexing_tree;
	TrackerMonitor *monitor;
	TrackerStorage *storage;
	TrackerConfig *config;
	GSettings *extractor_settings;
	TrackerFilesInterface *files_interface;

	GDBusProxy *control_proxy;
	GCancellable *control_proxy_cancellable;
	GPtrArray *control_proxy_folders;

	GSList *config_recursive_directories;
	GSList *config_single_directories;

	guint volumes_changed_id;

	guint index_removable_devices : 1;
};

enum {
	PROP_0,
	PROP_INDEXING_TREE,
	PROP_STORAGE,
	PROP_FILES_INTERFACE,
	PROP_MONITOR,
	N_PROPS,
};

static GParamSpec *props[N_PROPS] = { 0, };

G_DEFINE_TYPE (TrackerController, tracker_controller, G_TYPE_OBJECT)

static void
tracker_controller_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
	TrackerController *controller = TRACKER_CONTROLLER (object);

	switch (prop_id) {
	case PROP_INDEXING_TREE:
		controller->indexing_tree = g_value_dup_object (value);
		break;
	case PROP_STORAGE:
		controller->storage = g_value_dup_object (value);
		break;
	case PROP_FILES_INTERFACE:
		controller->files_interface = g_value_dup_object (value);
		break;
	case PROP_MONITOR:
		controller->monitor = g_value_dup_object (value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

static void
tracker_controller_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
	TrackerController *controller = TRACKER_CONTROLLER (object);

	switch (prop_id) {
	case PROP_INDEXING_TREE:
		g_value_set_object (value, controller->indexing_tree);
		break;
	case PROP_STORAGE:
		g_value_set_object (value, controller->storage);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

static void
add_indexed_directory (TrackerController     *controller,
                       GFile                 *file,
                       TrackerDirectoryFlags  flags)
{
	g_autofree gchar *path = NULL;
	TrackerStorageType type;

	path = g_file_get_path (file);

	if (path) {
		/* Do some simple checks for silly locations */
		if (strcmp (path, "/dev") == 0 ||
		    strcmp (path, "/lib") == 0 ||
		    strcmp (path, "/proc") == 0 ||
		    strcmp (path, "/sys") == 0) {
			return;
		}

		if (g_str_has_prefix (path, g_get_tmp_dir ())) {
			return;
		}
	}

	type = tracker_storage_get_type_for_file (controller->storage, file);

	if ((type & TRACKER_STORAGE_REMOVABLE) != 0)
		flags |= TRACKER_DIRECTORY_FLAG_IS_VOLUME;

	g_debug ("  Adding:'%s'", path);

	tracker_indexing_tree_add (controller->indexing_tree, file, flags);
}

static void
add_removable_directory (TrackerController *controller,
                         GFile             *mount_file)
{
	TrackerDirectoryFlags flags;
	g_autofree gchar *uri = NULL;

	flags = TRACKER_DIRECTORY_FLAG_RECURSE |
		TRACKER_DIRECTORY_FLAG_PRESERVE |
		TRACKER_DIRECTORY_FLAG_PRIORITY |
		TRACKER_DIRECTORY_FLAG_IS_VOLUME;

	uri = g_file_get_uri (mount_file);
	g_debug ("  Adding removable: '%s'", uri);
	tracker_indexing_tree_add (controller->indexing_tree,
				   mount_file,
				   flags);
}

static void
mount_point_added_cb (TrackerController *controller,
                      const gchar       *uuid,
                      const gchar       *mount_point,
                      const gchar       *mount_name,
                      gboolean           removable,
                      gboolean           optical,
                      TrackerStorage    *storage)
{
	g_autoptr (GFile) mount_point_file = NULL;

	g_debug ("Mount point added for path '%s'", mount_point);
	mount_point_file = g_file_new_for_path (mount_point);

	if (removable && !controller->index_removable_devices) {
		g_debug ("  Not crawling, removable devices disabled in config");
	} else if (!removable) {
		TrackerDirectoryFlags flags;
		GSList *l;

		/* Check if one of the recursively indexed locations is in
		 *   the mounted path, or if the mounted path is inside
		 *   a recursively indexed directory... */
		for (l = tracker_config_get_index_recursive_directories (controller->config);
		     l;
		     l = g_slist_next (l)) {
			g_autoptr (GFile) config_file = NULL;

			config_file = g_file_new_for_path (l->data);
			flags = TRACKER_DIRECTORY_FLAG_RECURSE |
				TRACKER_DIRECTORY_FLAG_PRESERVE;

			if (g_file_equal (config_file, mount_point_file) ||
			    g_file_has_prefix (config_file, mount_point_file)) {
				/* If the config path is contained inside the mount path,
				 *  then add the config path to re-check */
				g_debug ("  Re-check of configured path '%s' needed (recursively)",
				         (gchar *) l->data);
				add_indexed_directory (controller,
				                       config_file,
				                       flags);
			} else if (g_file_has_prefix (mount_point_file, config_file)) {
				/* If the mount path is contained inside the config path,
				 *  then add the mount path to re-check */
				g_debug ("  Re-check of path '%s' needed (inside configured path '%s')",
				         mount_point,
				         (gchar *) l->data);
				add_indexed_directory (controller,
				                       config_file,
				                       flags);
			}
		}

		/* Check if one of the non-recursively indexed locations is in
		 *  the mount path... */
		for (l = tracker_config_get_index_single_directories (controller->config);
		     l;
		     l = g_slist_next (l)) {
			g_autoptr (GFile) config_file = NULL;

			flags = TRACKER_DIRECTORY_FLAG_NONE;

			config_file = g_file_new_for_path (l->data);
			if (g_file_equal (config_file, mount_point_file) ||
			    g_file_has_prefix (config_file, mount_point_file)) {
				g_debug ("  Re-check of configured path '%s' needed (non-recursively)",
				         (gchar *) l->data);
				add_indexed_directory (controller,
				                       config_file,
				                       flags);
			}
		}
	} else {
		g_debug ("  Adding directories in removable media to crawler's queue");
		add_removable_directory (controller, mount_point_file);
	}
}

static void
mount_point_removed_cb (TrackerController *controller,
                        const gchar       *uuid,
                        const gchar       *mount_point,
                        TrackerStorage    *storage)
{
	g_autoptr (GFile) mount_point_file = NULL;

	g_debug ("Mount point removed for path '%s'", mount_point);

	mount_point_file = g_file_new_for_path (mount_point);
	tracker_indexing_tree_remove (controller->indexing_tree, mount_point_file);
}

static void
indexing_tree_update_filter (TrackerIndexingTree *indexing_tree,
			     TrackerFilterType    filter,
                             GStrv                strv)
{
	int i;

	tracker_indexing_tree_clear_filters (indexing_tree, filter);

	for (i = 0; strv[i]; i++)
		tracker_indexing_tree_add_filter (indexing_tree, filter, strv[i]);
}

static void
text_allowlist_update (TrackerIndexingTree *indexing_tree,
                       GStrv                strv)
{
	int i;

	tracker_indexing_tree_clear_allowed_text_patterns (indexing_tree);

	for (i = 0; strv[i]; i++)
		tracker_indexing_tree_add_allowed_text_pattern (indexing_tree, strv[i]);
}

static void
update_filters (TrackerController *controller)
{
	GStrv strv;

	/* Always ignore hidden */
	tracker_indexing_tree_set_filter_hidden (controller->indexing_tree, TRUE);

	/* Ignored files */
	strv = g_settings_get_strv (G_SETTINGS (controller->config), "ignored-files");
	indexing_tree_update_filter (controller->indexing_tree, TRACKER_FILTER_FILE, strv);
	g_strfreev (strv);

	/* Ignored directories */
	strv = g_settings_get_strv (G_SETTINGS (controller->config), "ignored-directories");
	indexing_tree_update_filter (controller->indexing_tree, TRACKER_FILTER_DIRECTORY, strv);
	g_strfreev (strv);

	/* Directories with content */
	strv = g_settings_get_strv (G_SETTINGS (controller->config), "ignored-directories-with-content");
	indexing_tree_update_filter (controller->indexing_tree, TRACKER_FILTER_PARENT_DIRECTORY, strv);
	g_strfreev (strv);

	/* Allowed text patterns */
	strv = g_settings_get_strv (controller->extractor_settings, "text-allowlist");
	text_allowlist_update (controller->indexing_tree, strv);
	g_strfreev (strv);
}

static void
update_directories_from_new_config (TrackerController *controller,
                                    GSList            *new_dirs,
                                    GSList            *old_dirs,
                                    gboolean           recurse)
{
	TrackerDirectoryFlags flags = 0;
	GSList *sl;

	TRACKER_NOTE (CONFIG, g_message ("Updating %s directories changed from configuration",
	                      recurse ? "recursive" : "single"));

	/* First remove all directories removed from the config */
	for (sl = old_dirs; sl; sl = sl->next) {
		const gchar *path;

		path = sl->data;

		/* If we are not still in the list, remove the dir */
		if (!tracker_string_in_gslist (path, new_dirs)) {
			g_autoptr (GFile) file = NULL;

			TRACKER_NOTE (CONFIG, g_message ("  Removing directory: '%s'", path));

			file = g_file_new_for_path (path);

			/* First, remove the preserve flag, it might be
			 * set on configuration directories within mount
			 * points, as data should be persistent across
			 * unmounts.
			 */
			tracker_indexing_tree_get_root (controller->indexing_tree,
			                                file, NULL, &flags);

			if ((flags & TRACKER_DIRECTORY_FLAG_PRESERVE) != 0) {
				flags &= ~(TRACKER_DIRECTORY_FLAG_PRESERVE);
				tracker_indexing_tree_add (controller->indexing_tree,
							   file, flags);
			}

			/* Fully remove item (monitors and from store),
			 * now that there's no preserve flag.
			 */
			tracker_indexing_tree_remove (controller->indexing_tree, file);
		}
	}

	flags = TRACKER_DIRECTORY_FLAG_NONE;

	if (recurse) {
		flags |= TRACKER_DIRECTORY_FLAG_RECURSE;
	}

	/* Second add directories which are new */
	for (sl = new_dirs; sl; sl = sl->next) {
		const gchar *path;

		path = sl->data;

		/* If we are now in the list, add the dir */
		if (!tracker_string_in_gslist (path, old_dirs)) {
			g_autoptr (GFile) file = NULL;

			TRACKER_NOTE (CONFIG, g_message ("  Adding directory:'%s'", path));

			file = g_file_new_for_path (path);
			add_indexed_directory (controller, file, flags);
		}
	}
}

static void
index_recursive_directories_cb (TrackerConfig     *config,
                                GParamSpec        *pspec,
                                TrackerController *controller)
{
	GSList *new_dirs, *old_dirs;

	new_dirs = tracker_config_get_index_recursive_directories (controller->config);
	old_dirs = controller->config_recursive_directories;

	update_directories_from_new_config (controller,
	                                    new_dirs,
	                                    old_dirs,
	                                    TRUE);

	g_slist_free_full (controller->config_recursive_directories, g_free);
	controller->config_recursive_directories = tracker_gslist_copy_with_string_data (new_dirs);
}

static void
index_single_directories_cb (TrackerConfig     *config,
                             GParamSpec        *pspec,
                             TrackerController *controller)
{
	GSList *new_dirs, *old_dirs;

	new_dirs = tracker_config_get_index_single_directories (config);
	old_dirs = controller->config_single_directories;

	update_directories_from_new_config (controller,
	                                    new_dirs,
	                                    old_dirs,
	                                    FALSE);

	g_slist_free_full (controller->config_single_directories, g_free);
	controller->config_single_directories = tracker_gslist_copy_with_string_data (new_dirs);
}

static void
filter_changed_cb (GSettings         *config,
                   GParamSpec        *pspec,
                   TrackerController *controller)
{
	update_filters (controller);
	tracker_indexing_tree_update_all (controller->indexing_tree);
}

static void
text_allowlist_changed_cb (TrackerConfig *config,
                           GParamSpec        *pspec,
                           TrackerController *controller)
{
	tracker_indexing_tree_update_all (controller->indexing_tree);
}

static gboolean
index_volumes_changed_idle (gpointer user_data)
{
	TrackerController *controller = user_data;
	GSList *mounts_removed = NULL;
	GSList *mounts_added = NULL;
	gboolean new_index_removable_devices;

	TRACKER_NOTE (CONFIG, g_message ("Volume related configuration changed, updating..."));

	new_index_removable_devices = g_settings_get_boolean (G_SETTINGS (controller->config), "index-removable-devices");

	/* Removable devices config changed? */
	if (controller->index_removable_devices != new_index_removable_devices) {
		GSList *m;

		m = tracker_storage_get_device_roots (controller->storage,
		                                      TRACKER_STORAGE_REMOVABLE,
		                                      TRUE);
		/* Set new config value */
		controller->index_removable_devices = new_index_removable_devices;

		if (controller->index_removable_devices) {
			/* If previously not indexing and now indexing, need to re-check
			 * current mounted volumes, add new monitors and index new files
			 */
			mounts_added = m;
		} else {
			/* If previously indexing and now not indexing, need to re-check
			 * current mounted volumes, remove monitors and remove all resources
			 * from the store belonging to a removable device
			 */
			mounts_removed = m;
		}
	}

	if (mounts_removed) {
		GSList *sl;

		for (sl = mounts_removed; sl; sl = g_slist_next (sl)) {
			g_autoptr (GFile) mount_point_file = NULL;

			mount_point_file = g_file_new_for_path (sl->data);
			tracker_indexing_tree_remove (controller->indexing_tree,
						      mount_point_file);
		}

		g_slist_free_full (mounts_removed, g_free);
	}

	if (mounts_added) {
		GSList *sl;

		for (sl = mounts_added; sl; sl = g_slist_next (sl)) {
			g_autoptr (GFile) mount_point_file = NULL;

			mount_point_file = g_file_new_for_path (sl->data);
			add_removable_directory (controller, mount_point_file);
		}

		g_slist_free_full (mounts_removed, g_free);
	}

	controller->volumes_changed_id = 0;

	return G_SOURCE_REMOVE;
}

static void
index_volumes_changed_cb (TrackerConfig     *config,
                          GParamSpec        *pspec,
                          TrackerController *controller)
{
	if (controller->volumes_changed_id == 0) {
		/* Set idle so multiple changes in the config lead to one check */
		controller->volumes_changed_id =
			g_idle_add (index_volumes_changed_idle, controller);
	}
}

static void
enable_monitors_changed_cb (GSettings         *config,
                            GParamSpec        *pspec,
                            TrackerController *controller)
{
	gboolean enable;

	enable = g_settings_get_boolean (config, "enable-monitors");

	if (enable != tracker_monitor_get_enabled (controller->monitor)) {
		tracker_monitor_set_enabled (controller->monitor, enable);
		if (enable)
			filter_changed_cb (config, pspec, controller);
	}
}

static void
initialize_from_config (TrackerController *controller)
{
	GSList *mounts = NULL, *l;
	GSList *dirs;

	update_filters (controller);

	if (controller->monitor) {
		tracker_monitor_set_enabled (controller->monitor,
		                             g_settings_get_boolean (G_SETTINGS (controller->config),
		                                                     "enable-monitors"));
	}

	/* Setup mount points, we MUST have config set up before we
	 * init mount points because the config is used in that
	 * function.
	 */
	controller->index_removable_devices =
		g_settings_get_boolean (G_SETTINGS (controller->config), "index-removable-devices");

	if (controller->index_removable_devices) {
		/* Get list of roots for removable devices */
		mounts = tracker_storage_get_device_roots (controller->storage,
		                                           TRACKER_STORAGE_REMOVABLE,
		                                           TRUE);
	}

	TRACKER_NOTE (CONFIG, g_message ("Setting up directories to iterate from config (IndexSingleDirectory)"));

	dirs = tracker_config_get_index_single_directories (controller->config);
	controller->config_single_directories = tracker_gslist_copy_with_string_data (dirs);

	for (; dirs; dirs = dirs->next) {
		g_autoptr (GFile) file = NULL;

		if (g_slist_find_custom (mounts, dirs->data, (GCompareFunc) g_strcmp0)) {
			g_debug ("  Duplicate found:'%s' - same as removable device path",
			         (gchar*) dirs->data);
			continue;
		}

		file = g_file_new_for_path (dirs->data);
		add_indexed_directory (controller, file,
		                       TRACKER_DIRECTORY_FLAG_NONE);
	}

	TRACKER_NOTE (CONFIG, g_message ("Setting up directories to iterate from config (IndexRecursiveDirectory)"));

	dirs = tracker_config_get_index_recursive_directories (controller->config);
	controller->config_recursive_directories = tracker_gslist_copy_with_string_data (dirs);

	for (; dirs; dirs = dirs->next) {
		g_autoptr (GFile) file = NULL;

		if (g_slist_find_custom (mounts, dirs->data, (GCompareFunc) g_strcmp0)) {
			g_debug ("  Duplicate found:'%s' - same as removable device path",
			         (gchar*) dirs->data);
			continue;
		}

		file = g_file_new_for_path (dirs->data);
		add_indexed_directory (controller, file,
		                       TRACKER_DIRECTORY_FLAG_RECURSE);
	}

	/* Add mounts */
	TRACKER_NOTE (CONFIG, g_message ("Setting up directories to iterate from devices/discs"));

	for (l = mounts; l; l = l->next) {
		g_autoptr (GFile) mount_point = NULL;

		mount_point = g_file_new_for_path (l->data);
		add_removable_directory (controller, mount_point);
	}

	g_slist_free_full (mounts, g_free);
}

static void
update_indexed_files_from_proxy (TrackerController *controller,
                                 GDBusProxy        *proxy)
{
	TrackerIndexingTree *indexing_tree;
	const gchar **indexed_uris = NULL;
	GVariant *v;
	gint i;

	v = g_dbus_proxy_get_cached_property (proxy, "IndexedLocations");
	if (v)
		indexed_uris = g_variant_get_strv (v, NULL);

	indexing_tree = controller->indexing_tree;

	/* Remove folders no longer there */
	for (i = 0; i < controller->control_proxy_folders->len; i++) {
		GFile *file;
		g_autofree gchar *uri = NULL;

		file = g_ptr_array_index (controller->control_proxy_folders, i);
		uri = g_file_get_uri (file);

		if (!indexed_uris || !g_strv_contains (indexed_uris, uri)) {
			tracker_indexing_tree_remove (indexing_tree,
			                              file);
		}
	}

	for (i = 0; indexed_uris && indexed_uris[i]; i++) {
		g_autoptr (GFileInfo) file_info = NULL;
		g_autoptr (GFile) file = NULL;

		file = g_file_new_for_uri (indexed_uris[i]);
		if (g_ptr_array_find_with_equal_func (controller->control_proxy_folders,
		                                      file,
		                                      (GEqualFunc) g_file_equal,
		                                      NULL))
			continue;

		file_info = g_file_query_info (file,
		                               G_FILE_ATTRIBUTE_STANDARD_TYPE ","
		                               G_FILE_ATTRIBUTE_STANDARD_IS_HIDDEN,
		                               G_FILE_QUERY_INFO_NOFOLLOW_SYMLINKS,
		                               NULL, NULL);

		if (!file_info)
			continue;

		if (g_file_info_get_file_type (file_info) == G_FILE_TYPE_DIRECTORY) {
			if (!tracker_indexing_tree_file_is_indexable (indexing_tree,
			                                              file, file_info)) {
				add_indexed_directory (controller,
				                       file,
				                       TRACKER_DIRECTORY_FLAG_RECURSE);
				g_ptr_array_add (controller->control_proxy_folders,
				                 g_object_ref (file));
			} else {
				tracker_indexing_tree_notify_update (indexing_tree, file, TRUE);
			}
		} else {
			tracker_indexing_tree_notify_update (indexing_tree, file, FALSE);
		}
	}

	g_free (indexed_uris);
}

static void
update_priority_graphs_from_proxy (TrackerController *controller,
                                   GDBusProxy        *proxy)
{
	GVariant *value;

	value = g_dbus_proxy_get_cached_property (proxy, "Graphs");
	tracker_files_interface_set_priority_graphs (controller->files_interface, value);
}

static void
proxy_properties_changed_cb (GDBusProxy *proxy,
                             GVariant   *changed_properties,
                             GStrv       invalidated_properties,
                             gpointer    user_data)
{
	update_indexed_files_from_proxy (user_data, proxy);
	update_priority_graphs_from_proxy (user_data, proxy);
}

static void
on_control_proxy_ready (GObject      *source,
                        GAsyncResult *res,
                        gpointer      user_data)
{
	TrackerController *controller = user_data;
	GError *error = NULL;

	controller->control_proxy = g_dbus_proxy_new_finish (res, &error);
	if (error) {
		g_critical ("Could not set up proxy: %s", error->message);
		g_error_free (error);
		return;
	}

	g_signal_connect (controller->control_proxy, "g-properties-changed",
	                  G_CALLBACK (proxy_properties_changed_cb), controller);
	update_indexed_files_from_proxy (controller, controller->control_proxy);
}

static void
log_config (TrackerConfig *config)
{
#ifdef G_ENABLE_DEBUG
	if (TRACKER_DEBUG_CHECK (CONFIG)) {
		GSList *dirs, *l;

		g_message ("Indexer options:");

		dirs = tracker_config_get_index_recursive_directories (config);
		if (dirs)
			g_message ("  Recursive folders:");
		for (l = dirs; l; l = l->next)
			g_message ("    %s\n", (char*) l->data);

		dirs = tracker_config_get_index_recursive_directories (config);
		if (dirs)
			g_message ("  Non-recursive folders:");
		for (l = dirs; l; l = l->next)
			g_message ("    %s\n", (char*) l->data);

		g_message ("  Index removable volumes: %s",
		           g_settings_get_boolean (G_SETTINGS (config),
		                                   "index-removable-devices") ?
		           "on" : "off");
		g_message ("  Monitor directories: %s",
		           g_settings_get_boolean (G_SETTINGS (config),
		                                   "enable-monitors") ?
		           "on" : "off");
	}
#endif
}

static void
tracker_controller_constructed (GObject *object)
{
	TrackerController *controller = TRACKER_CONTROLLER (object);

	G_OBJECT_CLASS (tracker_controller_parent_class)->constructed (object);

	g_signal_connect_object (controller->storage,
	                         "mount-point-added",
	                         G_CALLBACK (mount_point_added_cb),
	                         object,
	                         G_CONNECT_SWAPPED);
	g_signal_connect_object (controller->storage,
	                         "mount-point-removed",
	                         G_CALLBACK (mount_point_removed_cb),
	                         object,
	                         G_CONNECT_SWAPPED);

	controller->config = tracker_config_new ();
	g_signal_connect (controller->config, "changed::index-recursive-directories",
	                  G_CALLBACK (index_recursive_directories_cb),
	                  object);
	g_signal_connect (controller->config, "changed::index-single-directories",
	                  G_CALLBACK (index_single_directories_cb),
	                  object);
	g_signal_connect (controller->config, "changed::ignored-directories",
	                  G_CALLBACK (filter_changed_cb),
	                  object);
	g_signal_connect (controller->config, "changed::ignored-directories-with-content",
	                  G_CALLBACK (filter_changed_cb),
	                  object);
	g_signal_connect (controller->config, "changed::ignored-files",
	                  G_CALLBACK (filter_changed_cb),
	                  object);
	g_signal_connect (controller->config, "changed::enable-monitors",
	                  G_CALLBACK (enable_monitors_changed_cb),
	                  object);
	g_signal_connect (controller->config, "changed::index-removable-devices",
	                  G_CALLBACK (index_volumes_changed_cb),
	                  object);

	controller->extractor_settings = g_settings_new ("org.freedesktop.Tracker3.Extract");
	g_signal_connect (controller->extractor_settings, "changed::text-allowlist",
	                  G_CALLBACK (text_allowlist_changed_cb),
	                  object);

	g_dbus_proxy_new_for_bus (TRACKER_IPC_BUS,
	                          G_DBUS_PROXY_FLAGS_DO_NOT_AUTO_START,
	                          NULL,
	                          "org.freedesktop.LocalSearch3.Control",
	                          "/org/freedesktop/Tracker3/Miner/Files/Proxy",
	                          "org.freedesktop.Tracker3.Miner.Files.Proxy",
	                          NULL,
	                          on_control_proxy_ready,
	                          controller);

	initialize_from_config (controller);

	log_config (controller->config);
}

static void
tracker_controller_finalize (GObject *object)
{
	TrackerController *controller = TRACKER_CONTROLLER (object);

	g_cancellable_cancel (controller->control_proxy_cancellable);
	g_clear_object (&controller->control_proxy_cancellable);
	g_clear_object (&controller->control_proxy);
	g_clear_pointer (&controller->control_proxy_folders, g_ptr_array_unref);

	g_clear_handle_id (&controller->volumes_changed_id, g_source_remove);

	g_clear_object (&controller->indexing_tree);
	g_clear_object (&controller->storage);
	g_clear_object (&controller->config);
	g_clear_object (&controller->extractor_settings);
	g_clear_object (&controller->files_interface);

	g_slist_free_full (controller->config_single_directories, g_free);
	g_slist_free_full (controller->config_recursive_directories, g_free);

	G_OBJECT_CLASS (tracker_controller_parent_class)->finalize (object);
}

static void
tracker_controller_class_init (TrackerControllerClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS (klass);

	object_class->set_property = tracker_controller_set_property;
	object_class->get_property = tracker_controller_get_property;
	object_class->constructed = tracker_controller_constructed;
	object_class->finalize = tracker_controller_finalize;

	props[PROP_INDEXING_TREE] =
		g_param_spec_object ("indexing-tree",
		                     "Indexing tree",
		                     "Indexing tree",
		                     TRACKER_TYPE_INDEXING_TREE,
		                     G_PARAM_READWRITE |
		                     G_PARAM_CONSTRUCT_ONLY |
		                     G_PARAM_STATIC_STRINGS);
	props[PROP_STORAGE] =
		g_param_spec_object ("storage",
		                     "Storage",
		                     "Storage",
		                     TRACKER_TYPE_STORAGE,
		                     G_PARAM_READWRITE |
		                     G_PARAM_CONSTRUCT_ONLY |
		                     G_PARAM_STATIC_STRINGS);
	props[PROP_FILES_INTERFACE] =
		g_param_spec_object ("files-interface",
		                     NULL, NULL,
		                     TRACKER_TYPE_FILES_INTERFACE,
		                     G_PARAM_WRITABLE |
		                     G_PARAM_CONSTRUCT_ONLY |
		                     G_PARAM_STATIC_STRINGS);
	props[PROP_MONITOR] =
		g_param_spec_object ("monitor",
		                     NULL, NULL,
		                     TRACKER_TYPE_MONITOR,
		                     G_PARAM_WRITABLE |
		                     G_PARAM_CONSTRUCT_ONLY |
		                     G_PARAM_STATIC_STRINGS);

	g_object_class_install_properties (object_class, N_PROPS, props);
}

static void
tracker_controller_init (TrackerController *controller)
{
	controller->control_proxy_folders =
		g_ptr_array_new_with_free_func (g_object_unref);
}

TrackerController *
tracker_controller_new (TrackerIndexingTree   *tree,
                        TrackerMonitor        *monitor,
                        TrackerStorage        *storage,
                        TrackerFilesInterface *files_interface)
{
	return g_object_new (TRACKER_TYPE_CONTROLLER,
			     "indexing-tree", tree,
	                     "monitor", monitor,
	                     "storage", storage,
	                     "files-interface", files_interface,
			     NULL);
}
