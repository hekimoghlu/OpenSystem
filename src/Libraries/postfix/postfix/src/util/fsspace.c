/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
/* System library. */

#include <sys_defs.h>

#if defined(STATFS_IN_SYS_MOUNT_H)
#include <sys/param.h>
#include <sys/mount.h>
#elif defined(STATFS_IN_SYS_VFS_H)
#include <sys/vfs.h>
#elif defined(STATVFS_IN_SYS_STATVFS_H)
#include <sys/statvfs.h>
#elif defined(STATFS_IN_SYS_STATFS_H)
#include <sys/statfs.h>
#else
#ifdef USE_STATFS
#error "please specify the include file with `struct statfs'"
#else
#error "please specify the include file with `struct statvfs'"
#endif
#endif

/* Utility library. */

#include <msg.h>
#include <fsspace.h>

/* fsspace - find amount of available file system space */

void    fsspace(const char *path, struct fsspace * sp)
{
    const char *myname = "fsspace";

#ifdef USE_STATFS
#ifdef USE_STRUCT_FS_DATA			/* Ultrix */
    struct fs_data fsbuf;

    if (statfs(path, &fsbuf) < 0)
	msg_fatal("statfs %s: %m", path);
    sp->block_size = 1024;
    sp->block_free = fsbuf.fd_bfreen;
#else
    struct statfs fsbuf;

    if (statfs(path, &fsbuf) < 0)
	msg_fatal("statfs %s: %m", path);
    sp->block_size = fsbuf.f_bsize;
    sp->block_free = fsbuf.f_bavail;
#endif
#endif
#ifdef USE_STATVFS
    struct statvfs fsbuf;

    if (statvfs(path, &fsbuf) < 0)
	msg_fatal("statvfs %s: %m", path);
    sp->block_size = fsbuf.f_frsize;
    sp->block_free = fsbuf.f_bavail;
#endif
    if (msg_verbose)
	msg_info("%s: %s: block size %lu, blocks free %lu",
		 myname, path, sp->block_size, sp->block_free);
}

#ifdef TEST

 /*
  * Proof-of-concept test program: print free space unit and count for all
  * listed file systems.
  */

#include <vstream.h>

int     main(int argc, char **argv)
{
    struct fsspace sp;

    if (argc == 1)
	msg_fatal("usage: %s filesystem...", argv[0]);

    while (--argc && *++argv) {
	fsspace(*argv, &sp);
	vstream_printf("%10s: block size %lu, blocks free %lu\n",
		       *argv, sp.block_size, sp.block_free);
	vstream_fflush(VSTREAM_OUT);
    }
    return (0);
}

#endif
