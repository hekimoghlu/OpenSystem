/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#ifndef _FSTAB_H_
#define _FSTAB_H_

#include <_bounds.h>

_LIBC_SINGLE_BY_DEFAULT()

/*
 * File system table, see fstab(5).
 *
 * Used by dump, mount, umount, swapon, fsck, df, ...
 *
 * For ufs fs_spec field is the block special name.  Programs that want to
 * use the character special name must create that name by prepending a 'r'
 * after the right most slash.  Quota files are always named "quotas", so
 * if type is "rq", then use concatenation of fs_file and "quotas" to locate
 * quota file.
 */
#define	_PATH_FSTAB	"/etc/fstab"
#define	FSTAB		"/etc/fstab"	/* deprecated */

#define	FSTAB_RW	"rw"		/* read/write device */
#define	FSTAB_RQ	"rq"		/* read/write with quotas */
#define	FSTAB_RO	"ro"		/* read-only device */
#define	FSTAB_SW	"sw"		/* swap device */
#define	FSTAB_XX	"xx"		/* ignore totally */

struct fstab {
	char *_LIBC_CSTR	fs_spec;	/* block special device name */
	char *_LIBC_CSTR	fs_file;	/* file system path prefix */
	char *_LIBC_CSTR	fs_vfstype;	/* File system type, ufs, nfs */
	char *_LIBC_CSTR	fs_mntops;	/* Mount options ala -o */
	char *_LIBC_CSTR	fs_type;	/* FSTAB_* from fs_mntops */
	int			fs_freq;	/* dump frequency, in days */
	int			fs_passno;	/* pass number on parallel dump */
};

#include <sys/cdefs.h>

__BEGIN_DECLS
struct fstab *getfsent(void);
struct fstab *getfsspec(const char *);
struct fstab *getfsfile(const char *);
int setfsent(void);
void endfsent(void);
__END_DECLS

#endif /* !_FSTAB_H_ */
