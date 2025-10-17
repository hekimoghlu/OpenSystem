/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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
#include "archive_platform.h"

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#include "archive.h"
#include "archive_entry.h"

void
archive_entry_copy_stat(struct archive_entry *entry, const struct stat *st)
{
#if HAVE_STRUCT_STAT_ST_MTIMESPEC_TV_NSEC
	archive_entry_set_atime(entry, st->st_atime, st->st_atimespec.tv_nsec);
	archive_entry_set_ctime(entry, st->st_ctime, st->st_ctimespec.tv_nsec);
	archive_entry_set_mtime(entry, st->st_mtime, st->st_mtimespec.tv_nsec);
#elif HAVE_STRUCT_STAT_ST_MTIM_TV_NSEC
	archive_entry_set_atime(entry, st->st_atime, st->st_atim.tv_nsec);
	archive_entry_set_ctime(entry, st->st_ctime, st->st_ctim.tv_nsec);
	archive_entry_set_mtime(entry, st->st_mtime, st->st_mtim.tv_nsec);
#elif HAVE_STRUCT_STAT_ST_MTIME_NSEC
	archive_entry_set_atime(entry, st->st_atime, st->st_atime_nsec);
	archive_entry_set_ctime(entry, st->st_ctime, st->st_ctime_nsec);
	archive_entry_set_mtime(entry, st->st_mtime, st->st_mtime_nsec);
#elif HAVE_STRUCT_STAT_ST_MTIME_N
	archive_entry_set_atime(entry, st->st_atime, st->st_atime_n);
	archive_entry_set_ctime(entry, st->st_ctime, st->st_ctime_n);
	archive_entry_set_mtime(entry, st->st_mtime, st->st_mtime_n);
#elif HAVE_STRUCT_STAT_ST_UMTIME
	archive_entry_set_atime(entry, st->st_atime, st->st_uatime * 1000);
	archive_entry_set_ctime(entry, st->st_ctime, st->st_uctime * 1000);
	archive_entry_set_mtime(entry, st->st_mtime, st->st_umtime * 1000);
#elif HAVE_STRUCT_STAT_ST_MTIME_USEC
	archive_entry_set_atime(entry, st->st_atime, st->st_atime_usec * 1000);
	archive_entry_set_ctime(entry, st->st_ctime, st->st_ctime_usec * 1000);
	archive_entry_set_mtime(entry, st->st_mtime, st->st_mtime_usec * 1000);
#else
	archive_entry_set_atime(entry, st->st_atime, 0);
	archive_entry_set_ctime(entry, st->st_ctime, 0);
	archive_entry_set_mtime(entry, st->st_mtime, 0);
#endif
#if HAVE_STRUCT_STAT_ST_BIRTHTIMESPEC_TV_NSEC
	archive_entry_set_birthtime(entry, st->st_birthtime, st->st_birthtimespec.tv_nsec);
#elif HAVE_STRUCT_STAT_ST_BIRTHTIME
	archive_entry_set_birthtime(entry, st->st_birthtime, 0);
#else
	archive_entry_unset_birthtime(entry);
#endif
	archive_entry_set_dev(entry, st->st_dev);
	archive_entry_set_gid(entry, st->st_gid);
	archive_entry_set_uid(entry, st->st_uid);
	archive_entry_set_ino(entry, st->st_ino);
	archive_entry_set_nlink(entry, st->st_nlink);
	archive_entry_set_rdev(entry, st->st_rdev);
	archive_entry_set_size(entry, st->st_size);
	archive_entry_set_mode(entry, st->st_mode);
}
