/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef ARCHIVE_READ_DISK_PRIVATE_H_INCLUDED
#define ARCHIVE_READ_DISK_PRIVATE_H_INCLUDED

#ifndef __LIBARCHIVE_BUILD
#error This header is only to be used internally to libarchive.
#endif

#include "archive_platform_acl.h"

struct tree;
struct archive_entry;

struct archive_read_disk {
	struct archive	archive;

	/* Reused by archive_read_next_header() */
	struct archive_entry *entry;

	/*
	 * Symlink mode is one of 'L'ogical, 'P'hysical, or 'H'ybrid,
	 * following an old BSD convention.  'L' follows all symlinks,
	 * 'P' follows none, 'H' follows symlinks only for the first
	 * item.
	 */
	char	symlink_mode;

	/*
	 * Since symlink interaction changes, we need to track whether
	 * we're following symlinks for the current item.  'L' mode above
	 * sets this true, 'P' sets it false, 'H' changes it as we traverse.
	 */
	char	follow_symlinks;  /* Either 'L' or 'P'. */

	/* Directory traversals. */
	struct tree *tree;
	int	(*open_on_current_dir)(struct tree*, const char *, int);
	int	(*tree_current_dir_fd)(struct tree*);
	int	(*tree_enter_working_dir)(struct tree*);

	/* Bitfield with ARCHIVE_READDISK_* tunables */
	int	flags;

	const char * (*lookup_gname)(void *private, int64_t gid);
	void	(*cleanup_gname)(void *private);
	void	 *lookup_gname_data;
	const char * (*lookup_uname)(void *private, int64_t uid);
	void	(*cleanup_uname)(void *private);
	void	 *lookup_uname_data;

	int	(*metadata_filter_func)(struct archive *, void *,
			struct archive_entry *);
	void	*metadata_filter_data;

	/* ARCHIVE_MATCH object. */
	struct archive	*matching;
	/* Callback function, this will be invoked when ARCHIVE_MATCH
	 * archive_match_*_excluded_ae return true. */
	void	(*excluded_cb_func)(struct archive *, void *,
			 struct archive_entry *);
	void	*excluded_cb_data;
};

const char *
archive_read_disk_entry_setup_path(struct archive_read_disk *,
    struct archive_entry *, int *);

int
archive_read_disk_entry_setup_acls(struct archive_read_disk *,
    struct archive_entry *, int *);
#endif
