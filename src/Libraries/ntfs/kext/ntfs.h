/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#ifndef _OSX_NTFS_H
#define _OSX_NTFS_H

#ifdef KERNEL

#include <sys/mount.h>

#include <kern/locks.h>

/* The email address of the NTFS developers. */
__attribute__((visibility("hidden"))) extern const char ntfs_dev_email[];
__attribute__((visibility("hidden"))) extern const char ntfs_please_email[];

/*
 * Lock group and lock attribute for de-/initialization of locks (defined
 * in ntfs_vfsops.c).
 */
__attribute__((visibility("hidden"))) extern lck_grp_t *ntfs_lock_grp;
__attribute__((visibility("hidden"))) extern lck_attr_t *ntfs_lock_attr;

#include "ntfs_volume.h"

/**
 * NTFS_MP - return the NTFS volume given a vfs mount
 * @mp:		VFS mount
 *
 * NTFS_MP() returns the NTFS volume associated with the VFS mount @mp.
 */
static inline ntfs_volume *NTFS_MP(mount_t mp)
{
	return (ntfs_volume*)vfs_fsprivate(mp);
}

__private_extern__ void ntfs_do_postponed_release(ntfs_volume *vol);

#endif /* KERNEL */

#include "ntfs_endian.h"
#include "ntfs_types.h"

/* Some useful constants to do with NTFS. */
enum {
	NTFS_BLOCK_SIZE		= 512,
	NTFS_BLOCK_SIZE_SHIFT	= 9,
	NTFS_MAX_NAME_LEN	= 255,
	NTFS_MAX_ATTR_NAME_LEN	= 255,
	NTFS_MAX_SECTOR_SIZE	= 4096,		/* 4kiB */
	NTFS_MAX_CLUSTER_SIZE	= 64 * 1024,	/* 64kiB */
	NTFS_ALLOC_BLOCK	= 1024,
	NTFS_MAX_HARD_LINKS	= 65535,	/* 2^16 - 1 */
	NTFS_MAX_ATTR_LIST_SIZE	= 256 * 1024,	/* 256kiB, corresponding to the
						   VACB_MAPPING_GRANULARITY on
						   Windows. */
	NTFS_COMPRESSION_UNIT	= 4,
};

/*
 * The maximum attribute size on NTFS is 2^63 - 1 bytes as it is stored in a
 * signed 64 bit type (s64).
 */
#define NTFS_MAX_ATTRIBUTE_SIZE 0x7fffffffffffffffULL

/*
 * The maximum number of MFT records allowed on NTFS is 2^32 as described in
 * various documentation to be found on the Microsoft web site.  This is an
 * imposed limit rather than an inherent NTFS format limit.
 */
#define NTFS_MAX_NR_MFT_RECORDS 0x100000000ULL

#define NTFS_SUB_SECTOR_MFT_RECORD_SIZE_RW 1

// TODO: Constants so ntfs_vfsops.c compiles for now...
enum {
	/* One of these must be present, default is ON_ERRORS_CONTINUE|ON_ERRORS_FAIL_DIRTY. */
	ON_ERRORS_PANIC		= 0x01,
	ON_ERRORS_REMOUNT_RO	= 0x02,
	ON_ERRORS_CONTINUE	= 0x04,
	/* Optional, can be combined with any of the above. */
	ON_ERRORS_RECOVER	= 0x10,
	/* If the volume is dirty, and we attempted to mount read/write, */
	/* return an error rather than force a read-only mount. */
	ON_ERRORS_FAIL_DIRTY    = 0x20,
};

/*
 * The NTFS mount options header passed in from user space.
 */
typedef struct {
#ifndef KERNEL
	char *fspec;	/* Path of device to mount, consumed by mount(2). */
#endif /* !KERNEL */
	u8 major_ver;	/* The major version of the mount options structure. */
	u8 minor_ver;	/* The minor version of the mount options structure. */
} __attribute__((__packed__)) ntfs_mount_options_header;

/*
 * The NTFS mount options passed in from user space.  This follows the
 * ntfs_mount_options_header aligned to an eight byte boundary.
 *
 * This is major version 0, minor version 0, which does not have any options,
 * i.e. is empty.
 */
typedef struct {
	/* Mount options version 0.0 does not have any ntfs options. */
} __attribute__((__packed__)) ntfs_mount_options_0_0;

/*
 * The currently defined flags for the ntfs mount options structure.
 */
enum {
	/* Below flag(s) appeared in mount options version 1.0. */
	NTFS_MNT_OPT_CASE_SENSITIVE = const_cpu_to_le32(0x00000001),
	/* Below flag(s) appeared in mount options version x.y. */
	// TODO: Add NTFS specific mount options flags here.
};

typedef le32 NTFS_MNT_OPTS;

/*
 * The NTFS mount options passed in from user space.  This follows the
 * ntfs_mount_options_header aligned to an eight byte boundary.
 *
 * This is major version 1, minor version 0, which has only one option, a
 * little endian, 32-bit flags option.
 */
typedef struct {
	NTFS_MNT_OPTS flags;
	// TODO: Add NTFS specific mount options here.
} __attribute__((__packed__)) ntfs_mount_options_1_0;

#endif /* !_OSX_NTFS_H */
