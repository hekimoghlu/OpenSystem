/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 26, 2025.
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
/*
 * 26-Oct-2004
 * DRI: Rob Braun <bbraun@synack.net>
 * Ported from xar-unsaxy 16-Apr-2005
 */
/*
 * Portions Copyright 2006, Apple Computer, Inc.
 * Christopher Ryan <ryanc@apple.com>
*/

#include "config.h"
#ifndef HAVE_ASPRINTF
#include "asprintf.h"
#endif
#include <stdio.h>
#include <unistd.h>
#include "xar.h"
#include "arcmod.h"
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include "ext2.h"

#ifdef HAVE_EXT2FS_EXT2_FS_H
#include <ext2fs/ext2_fs.h>
#else
#if defined(HAVE_LINUX_EXT2_FS_H)
typedef uint32_t u32;
typedef uint8_t u8;
#include <linux/ext2_fs.h>
#endif
#endif

#define XAR_EXT2_FORK "ext2"

#if defined(HAVE_EXT2FS_EXT2_FS_H) || defined(HAVE_LINUX_EXT2_FS_H)
static void x_addprop(xar_file_t f, const char *name) {
	char opt[1024];
	memset(opt, 0, sizeof(opt));
	snprintf(opt, sizeof(opt)-1, "%s/%s", XAR_ATTR_FORK, name);
	xar_prop_set(f, opt, NULL);
	xar_attr_set(f, opt, "fstype", "ext2");
	return;
}
#endif

int xar_ext2attr_archive(xar_t x, xar_file_t f, const char* file, const char *buffer, size_t len)
{
	int ret = 0;
	
	/* if archiving from a buffer, then there is no place to get extattr */
	if ( len )
		return 0;
		
#if defined(HAVE_EXT2FS_EXT2_FS_H) || defined(HAVE_LINUX_EXT2_FS_H)
	int fd, flags=0, version;
	char *vstr;
	const char *opt;

        xar_prop_get(f, "type", &opt);
        if(!opt) return 0;
        if( strcmp(opt, "file") != 0 ) {
                if( strcmp(opt, "hardlink") != 0 )
                	if( strcmp(opt, "directory") != 0 )
				return 0;
	}

	fd = open(file, O_RDONLY);
	if( fd < 0 ) {
		return 0;
	}
	if( ioctl(fd, EXT2_IOC_GETVERSION, &version) < 0 ) {
		ret = 0;
		goto BAIL;
	}
	if( ioctl(fd, EXT2_IOC_GETFLAGS, &flags) < 0 ) {
		ret = 0;
		goto BAIL;
	}

	if( flags == 0 ) goto BAIL;

	xar_prop_set(f, XAR_EXT2_FORK, NULL);
	asprintf(&vstr, "%d", version);
	xar_attr_set(f, XAR_EXT2_FORK, "version", vstr);
	free(vstr);

	if(! (flags & ~EXT2_SECRM_FL) )
		x_addprop(f, "SecureDeletion");
	if(! (flags & ~EXT2_UNRM_FL) )
		x_addprop(f, "Undelete");
	if(! (flags & ~EXT2_COMPR_FL) )
		x_addprop(f, "Compress");
	if(! (flags & ~EXT2_SYNC_FL) )
		x_addprop(f, "Synchronous");
	if(! (flags & ~EXT2_IMMUTABLE_FL) )
		x_addprop(f, "Immutable");
	if(! (flags & ~EXT2_APPEND_FL) )
		x_addprop(f, "AppendOnly");
	if(! (flags & ~EXT2_NODUMP_FL) )
		x_addprop(f, "NoDump");
	if(! (flags & ~EXT2_NOATIME_FL) )
		x_addprop(f, "NoAtime");
	if(! (flags & ~EXT2_DIRTY_FL) )
		x_addprop(f, "CompDirty");
	if(! (flags & ~EXT2_COMPRBLK_FL) )
		x_addprop(f, "CompBlock");
#ifdef EXT2_NOCOMPR_FL
	if(! (flags & ~EXT2_NOCOMPR_FL) )
		x_addprop(f, "NoCompBlock");
#endif
	if(! (flags & ~EXT2_ECOMPR_FL) )
		x_addprop(f, "CompError");
	if(! (flags & ~EXT2_BTREE_FL) )
		x_addprop(f, "BTree");
	if(! (flags & ~EXT2_INDEX_FL) )
		x_addprop(f, "HashIndexed");
	if(! (flags & ~EXT2_IMAGIC_FL) )
		x_addprop(f, "iMagic");
#ifdef EXT3_JOURNAL_DATA_FL
	if(! (flags & ~EXT3_JOURNAL_DATA_FL) )
		x_addprop(f, "Journaled");
#endif
	if(! (flags & ~EXT2_NOTAIL_FL) )
		x_addprop(f, "NoTail");
	if(! (flags & ~EXT2_DIRSYNC_FL) )
		x_addprop(f, "DirSync");
	if(! (flags & ~EXT2_TOPDIR_FL) )
		x_addprop(f, "TopDir");
	if(! (flags & ~EXT2_RESERVED_FL) )
		x_addprop(f, "Reserved");

BAIL:
	close(fd);
#endif
	return ret;
}

#if defined(HAVE_EXT2FS_EXT2_FS_H) || defined(HAVE_LINUX_EXT2_FS_H)
static int32_t e2prop_get(xar_file_t f, const char *name, char **value) {
	char v[1024];

	memset(v, 0, sizeof(v));
	snprintf(v, sizeof(v)-1, "%s/%s", XAR_ATTR_FORK, name);
	return xar_prop_get(f, v, (const char**)value);
}
#endif

int xar_ext2attr_extract(xar_t x, xar_file_t f, const char* file, char *buffer, size_t len)
{
	/* if extracting to a buffer, then there is no place to write extattr */
	if ( len )
		return 0;
	
#if defined(HAVE_EXT2FS_EXT2_FS_H) || defined(HAVE_LINUX_EXT2_FS_H)
	int fd = -1, version, flags = 0;
	char *tmp;

	if( xar_prop_get(f, XAR_EXT2_FORK, NULL) == 0 ) {
		const char *temp;
		temp = xar_attr_get(f, XAR_EXT2_FORK, "version");
		version = strtol(temp, NULL, 10);
		fd = open(file, O_RDONLY);
		if( fd < 0 )
			return 0;
		ioctl(fd, EXT2_IOC_SETVERSION, &version);
	}

	if( xar_prop_get(f, XAR_ATTR_FORK, NULL)  ) {
		if( fd >= 0 ) close(fd);
		return 0;
	}

	if( e2prop_get(f, "SecureDeletion", (char **)&tmp) == 0 )
		flags |= EXT2_SECRM_FL;
	if( e2prop_get(f, "Undelete", (char **)&tmp) == 0 )
		flags |= EXT2_UNRM_FL ;
	if( e2prop_get(f, "Compress", (char **)&tmp) == 0 )
		flags |= EXT2_COMPR_FL ;
	if( e2prop_get(f, "Synchronous", (char **)&tmp) == 0 )
		flags |= EXT2_SYNC_FL ;
	if( e2prop_get(f, "SystemImmutable", (char **)&tmp) == 0 )
		flags |= EXT2_IMMUTABLE_FL ;
	if( e2prop_get(f, "AppendOnly", (char **)&tmp) == 0 )
		flags |= EXT2_APPEND_FL ;
	if( e2prop_get(f, "NoDump", (char **)&tmp) == 0 )
		flags |= EXT2_NODUMP_FL ;
	if( e2prop_get(f, "NoAtime", (char **)&tmp) == 0 )
		flags |= EXT2_NOATIME_FL ;
	if( e2prop_get(f, "CompDirty", (char **)&tmp) == 0 )
		flags |= EXT2_DIRTY_FL ;
	if( e2prop_get(f, "CompBlock", (char **)&tmp) == 0 )
		flags |= EXT2_COMPRBLK_FL ;
#ifdef EXT2_NOCOMPR_FL
	if( e2prop_get(f, "NoCompBlock", (char **)&tmp) == 0 )
		flags |= EXT2_NOCOMPR_FL ;
#endif
	if( e2prop_get(f, "CompError", (char **)&tmp) == 0 )
		flags |= EXT2_ECOMPR_FL ;
	if( e2prop_get(f, "BTree", (char **)&tmp) == 0 )
		flags |= EXT2_BTREE_FL ;
	if( e2prop_get(f, "HashIndexed", (char **)&tmp) == 0 )
		flags |= EXT2_INDEX_FL ;
	if( e2prop_get(f, "iMagic", (char **)&tmp) == 0 )
		flags |= EXT2_IMAGIC_FL ;
#ifdef EXT3_JOURNAL_DATA_FL
	if( e2prop_get(f, "Journaled", (char **)&tmp) == 0 )
		flags |= EXT3_JOURNAL_DATA_FL ;
#endif
	if( e2prop_get(f, "NoTail", (char **)&tmp) == 0 )
		flags |= EXT2_NOTAIL_FL ;
	if( e2prop_get(f, "DirSync", (char **)&tmp) == 0 )
		flags |= EXT2_DIRSYNC_FL ;
	if( e2prop_get(f, "TopDir", (char **)&tmp) == 0 )
		flags |= EXT2_TOPDIR_FL ;

	if( fd < 0 ) {
		fd = open(file, O_RDONLY);
		if( fd < 0 )
			return 0;
	}

	ioctl(fd, EXT2_IOC_SETFLAGS, &flags);
	close(fd);
#endif
	return 0;
}
