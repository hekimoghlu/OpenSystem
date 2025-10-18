/*
 * Copyright (C) 2023, Red Hat Inc.
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
 *
 * Author: Carlos Garnacho <carlosg@gnome.org>
 */

#include "config-miners.h"

#include "tracker-landlock.h"

#include <glib/gstdio.h>
#include <fcntl.h>
#include <linux/landlock.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "valgrind.h"
#include "tracker-debug.h"

/* Compensate for these syscalls not being wrapped in libc */
#define CREATE_RULESET(attr, flags) \
	syscall (SYS_landlock_create_ruleset, (attr), \
	         (attr) != NULL ? sizeof *(attr) : 0, \
	         flags)

#define ADD_RULE(fd, type, attr, flags) \
	syscall (SYS_landlock_add_rule, fd, type, (attr), flags)

#define RESTRICT_SELF(fd, flags) \
	syscall (SYS_landlock_restrict_self, fd, flags)

/* Cautious defines of flags in newer ABI versions */
#ifndef LANDLOCK_ACCESS_FS_REFER
#define LANDLOCK_ACCESS_FS_REFER (1ULL << 13)
#endif

#ifndef LANDLOCK_ACCESS_FS_TRUNCATE
#define LANDLOCK_ACCESS_FS_TRUNCATE (1ULL << 14)
#endif

typedef struct _TrackerLandlockRule TrackerLandlockRule;

struct _TrackerLandlockRule
{
	const gchar *path;
	guint64 flags;
};

gboolean
get_supported_fs_flags (guint32 *flags_out)
{
	guint64 supported_abi_flags[] = {
		/* Version 1 */
		(LANDLOCK_ACCESS_FS_EXECUTE |
		 LANDLOCK_ACCESS_FS_WRITE_FILE |
		 LANDLOCK_ACCESS_FS_READ_FILE |
		 LANDLOCK_ACCESS_FS_READ_DIR |
		 LANDLOCK_ACCESS_FS_REMOVE_DIR |
		 LANDLOCK_ACCESS_FS_REMOVE_FILE |
		 LANDLOCK_ACCESS_FS_MAKE_CHAR |
		 LANDLOCK_ACCESS_FS_MAKE_DIR |
		 LANDLOCK_ACCESS_FS_MAKE_REG |
		 LANDLOCK_ACCESS_FS_MAKE_SOCK |
		 LANDLOCK_ACCESS_FS_MAKE_FIFO |
		 LANDLOCK_ACCESS_FS_MAKE_BLOCK |
		 LANDLOCK_ACCESS_FS_MAKE_SYM),
		/* Version 2 */
		LANDLOCK_ACCESS_FS_REFER,
		/* Version 3 */
		LANDLOCK_ACCESS_FS_TRUNCATE,
	};
	guint32 flags = 0;
	int i, abi;

	abi = CREATE_RULESET (NULL, LANDLOCK_CREATE_RULESET_VERSION);
	if (abi < 0) {
		g_critical ("Could not get landlock supported ABI: %m");
		return FALSE;
	}

	for (i = 0; i < MIN (G_N_ELEMENTS (supported_abi_flags), abi); i++)
		flags |= supported_abi_flags[i];

	*flags_out = flags;

	return TRUE;
}

static void
add_rule (int          landlock_fd,
          const gchar *path,
          guint64      flags)
{
	struct landlock_path_beneath_attr attr = { 0, };
	int fd;
	int result;

	TRACKER_NOTE (SANDBOX, g_message ("Adding Landlock rule for '%s', flags %" G_GINT64_MODIFIER "x", path, flags));

	if (!g_file_test (path, G_FILE_TEST_EXISTS)) {
		g_debug ("Path %s does not exist in filesystem", path);
		return;
	}

	fd = open (path, O_PATH | O_CLOEXEC);
	if (fd < 0) {
		g_warning ("Could not open '%s' to apply landlock rules: %m", path);
		return;
	}

	attr = (struct landlock_path_beneath_attr) {
		.allowed_access = flags,
		.parent_fd = fd,
	};

	result = ADD_RULE (landlock_fd, LANDLOCK_RULE_PATH_BENEATH, &attr, 0);
	close(fd);
	if (result != 0) {
		g_warning ("Could not add landlock rule for '%s': %m", path);
		return;
	}
}

static gboolean
create_ruleset (int *landlock_fd)
{
	struct landlock_ruleset_attr attr;
	int fd;
	guint32 flags;

	/* Get supported flags per the landlock ABI available */
	if (!get_supported_fs_flags (&flags))
		return FALSE;

	/* Create ruleset */
	attr = (struct landlock_ruleset_attr) {
		.handled_access_fs = flags,
	};

	fd = CREATE_RULESET (&attr, 0);
	if (fd < 0) {
		g_critical ("Failed to create landlock ruleset: %m");
		return FALSE;
	}

	*landlock_fd = fd;

	return TRUE;
}

static gboolean
apply_ruleset (int landlock_fd)
{
	/* Restrict any future new permission, necessary for the next step */
	if (prctl (PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0) {
		g_critical ("Failed to restrict new privileges: %m");
		return FALSE;
	}

	if (RESTRICT_SELF (landlock_fd, 0) != 0) {
		g_critical ("Failed to apply landlock ruleset: %m");
		return FALSE;
	}

	return TRUE;
}

gboolean
tracker_landlock_init (const gchar * const *indexed_folders)
{
	TrackerLandlockRule stock_rules[] = {
		/* Allow access to the executable itself */
		{ LIBEXECDIR "/localsearch-extractor-3",
		  LANDLOCK_ACCESS_FS_READ_FILE |
		  LANDLOCK_ACCESS_FS_EXECUTE },
		/* Library dirs, as we shockingly use libraries. Extends to /usr */
		{ PREFIX "/" LIBDIR,
		  (LANDLOCK_ACCESS_FS_EXECUTE |
		   LANDLOCK_ACCESS_FS_READ_FILE |
		   LANDLOCK_ACCESS_FS_READ_DIR) },
#if INTPTR_MAX == INT64_MAX
		{ "/usr/lib64",
		  (LANDLOCK_ACCESS_FS_EXECUTE |
		   LANDLOCK_ACCESS_FS_READ_FILE |
		   LANDLOCK_ACCESS_FS_READ_DIR) },
#endif
		{ "/usr/lib",
		  (LANDLOCK_ACCESS_FS_EXECUTE |
		   LANDLOCK_ACCESS_FS_READ_FILE |
		   LANDLOCK_ACCESS_FS_READ_DIR) },
		/* Required for the rtld on non-usrmerge systems */
		{ "/lib",
		  (LANDLOCK_ACCESS_FS_EXECUTE |
		   LANDLOCK_ACCESS_FS_READ_FILE |
		   LANDLOCK_ACCESS_FS_READ_DIR) },
		/* Data dir, to access miscellaneous files. Extends to /usr */
		{ PREFIX "/" DATADIR,
		  (LANDLOCK_ACCESS_FS_READ_FILE |
		   LANDLOCK_ACCESS_FS_READ_DIR) },
		{ "/usr/share",
		  (LANDLOCK_ACCESS_FS_READ_FILE |
		   LANDLOCK_ACCESS_FS_READ_DIR) },
		/* Necessary for libosinfo in Ubuntu/Debian */
		{ "/var/lib/usbutils",
		  LANDLOCK_ACCESS_FS_READ_FILE },
		/* Necessary for g_get_user_name() */
		{ "/etc/passwd",
		  LANDLOCK_ACCESS_FS_READ_FILE },
		/* Necessary for fontconfig */
		{ "/etc/fonts/",
		  LANDLOCK_ACCESS_FS_READ_FILE |
		  LANDLOCK_ACCESS_FS_READ_DIR },
	};
	TrackerLandlockRule homedir_rules[] = {
		/* Disable file access to sensitive folders the extractor has
		 * no business with. Since a flag bit needs to be set, only
		 * allow dir read access.
		 */
		{ ".ssh", LANDLOCK_ACCESS_FS_READ_DIR },
		{ ".pki", LANDLOCK_ACCESS_FS_READ_DIR },
		{ ".gnupg", LANDLOCK_ACCESS_FS_READ_DIR },
	};
	g_autofree gchar *current_dir = NULL, *cache_dir = NULL;
	g_auto (GStrv) library_paths = NULL;
	const gchar *ld_library_path = NULL;
	int i, landlock_fd;
	gboolean retval;

	if (RUNNING_ON_VALGRIND) {
		g_message ("Running under valgrind, Landlock was disabled");
		return TRUE;
	}

	if (!create_ruleset (&landlock_fd))
		return FALSE;

	/* Populate ruleset */
	for (i = 0; i < G_N_ELEMENTS (stock_rules); i++) {
		add_rule (landlock_fd, stock_rules[i].path,
		          stock_rules[i].flags);
	}

	for (i = 0; i < G_N_ELEMENTS (homedir_rules); i++) {
		g_autofree gchar *homedir_path = NULL;

		homedir_path = g_build_filename (g_get_home_dir (),
		                                 homedir_rules[i].path, NULL);
		add_rule (landlock_fd, homedir_path, homedir_rules[i].flags);
	}

	for (i = 0; indexed_folders && indexed_folders[i]; i++) {
		add_rule (landlock_fd,
		          indexed_folders[i],
		          LANDLOCK_ACCESS_FS_READ_FILE |
		          LANDLOCK_ACCESS_FS_READ_DIR);
	}

	/* Cater for development environments */
	ld_library_path = g_getenv ("LD_LIBRARY_PATH");
	if (ld_library_path) {
		library_paths = g_strsplit (ld_library_path, ":", -1);
		for (i = 0; library_paths && library_paths[i]; i++) {
			add_rule (landlock_fd,
			          library_paths[i],
			          LANDLOCK_ACCESS_FS_EXECUTE |
			          LANDLOCK_ACCESS_FS_READ_FILE |
			          LANDLOCK_ACCESS_FS_READ_DIR);
		}
	}

	current_dir = g_get_current_dir ();

	/* Detect running in-tree */
	if (g_strcmp0 (current_dir, BUILDROOT) == 0) {
		TrackerLandlockRule in_tree_rules[] = {
			{ BUILDROOT,
			  (LANDLOCK_ACCESS_FS_READ_FILE |
			   LANDLOCK_ACCESS_FS_READ_DIR |
			   LANDLOCK_ACCESS_FS_EXECUTE) },
			{ SRCROOT,
			  (LANDLOCK_ACCESS_FS_READ_FILE |
			   LANDLOCK_ACCESS_FS_READ_DIR) },
		};

		for (i = 0; i < G_N_ELEMENTS (in_tree_rules); i++) {
			add_rule (landlock_fd, in_tree_rules[i].path,
			          in_tree_rules[i].flags);
		}
	}

	/* Add user cache for readonly databases */
#ifdef MINER_FS_CACHE_LOCATION
	add_rule (landlock_fd, MINER_FS_CACHE_LOCATION,
	          LANDLOCK_ACCESS_FS_READ_FILE);
#else
	cache_dir = g_build_filename (g_get_user_cache_dir (), "tracker3", "files", NULL);
	add_rule (landlock_fd, cache_dir,
	          LANDLOCK_ACCESS_FS_READ_FILE);
#endif

	TRACKER_NOTE (SANDBOX, g_message ("Applying Landlock ruleset to PID %d", getpid ()));
	retval = apply_ruleset (landlock_fd);
	close (landlock_fd);

	return retval;
}
