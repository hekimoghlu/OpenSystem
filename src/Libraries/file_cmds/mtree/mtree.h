/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#ifndef _MTREE_H_
#define _MTREE_H_

#include <sys/time.h>
#include <string.h>
#include <stdlib.h>

#define	KEYDEFAULT \
	(F_GID | F_MODE | F_NLINK | F_SIZE | F_SLINK | F_TIME | F_UID | F_FLAGS)

#define	MISMATCHEXIT	2

typedef struct _node {
	struct _node	*parent, *child;	/* up, down */
	struct _node	*prev, *next;		/* left, right */
	off_t	st_size;			/* size */
	struct timespec	st_mtimespec;		/* last modification time */
	u_long	cksum;				/* check sum */
	char	*md5digest;			/* MD5 digest */
	char	*sha1digest;			/* SHA-1 digest */
	char	*sha256digest;			/* SHA-256 digest */
	char	*rmd160digest;			/* RIPEMD160 digest */
	char	*slink;				/* symbolic link reference */
	uid_t	st_uid;				/* uid */
	gid_t	st_gid;				/* gid */
#define	MBITS	(S_ISUID|S_ISGID|S_ISTXT|S_IRWXU|S_IRWXG|S_IRWXO)
	mode_t	st_mode;			/* mode */
	u_long	st_flags;			/* flags */
	nlink_t	st_nlink;			/* link count */
	struct timespec st_birthtimespec;	/* birth time (creation time) */
	struct timespec st_atimespec;		/* access time */
	struct timespec st_ctimespec;		/* metadata modification time */
	struct timespec st_ptimespec;		/* time added to parent folder */
	char	*xattrsdigest;			/* digest of extended attributes */
	u_quad_t xdstream_priv_id;		/* private id of the xattr data stream */
	ino_t	st_ino;				/* inode */
	char	*acldigest;			/* digest of access control list */
	u_quad_t  sibling_id;			/* sibling id */
	u_quad_t  nxattr;			/* xattr count */
	uint	protection_class;		/* data protection class */
	u_quad_t purgeable;			/* APFS purgeable flags */

#define	F_CKSUM			0x000000000001	/* check sum */
#define	F_DONE			0x000000000002	/* directory done */
#define	F_GID			0x000000000004	/* gid */
#define	F_GNAME			0x000000000008	/* group name */
#define	F_IGN			0x000000000010	/* ignore */
#define	F_MAGIC			0x000000000020	/* name has magic chars */
#define	F_MODE			0x000000000040	/* mode */
#define	F_NLINK			0x000000000080	/* number of links */
#define	F_SIZE			0x000000000100	/* size */
#define	F_SLINK			0x000000000200	/* The file the symbolic link is expected to reference */
#define	F_TIME			0x000000000400	/* modification time (mtime) */
#define	F_TYPE			0x000000000800	/* file type */
#define	F_UID			0x000000001000	/* uid */
#define	F_UNAME			0x000000002000	/* user name */
#define	F_VISIT			0x000000004000	/* file visited */
#define	F_MD5			0x000000008000	/* MD5 digest */
#define	F_NOCHANGE		0x000000010000	/* If owner/mode "wrong", do not change */
#define	F_SHA1			0x000000020000	/* SHA-1 digest */
#define	F_RMD160		0x000000040000	/* RIPEMD160 digest */
#define	F_FLAGS			0x000000080000	/* file flags */
#define	F_SHA256		0x000000100000	/* SHA-256 digest */
#define	F_BTIME			0x000000200000	/* creation time */
#define	F_ATIME			0x000000400000	/* access time */
#define	F_CTIME			0x000000800000	/* metadata modification time (ctime) */
#define	F_PTIME			0x000001000000	/* time added to parent folder */
#define	F_XATTRS		0x000002000000	/* digest of extended attributes */
#define	F_INODE			0x000004000000	/* inode */
#define	F_ACL			0x000008000000	/* digest of access control list */
#define	F_SIBLINGID		0x000010000000	/* sibling id */
#define	F_NXATTR		0x000020000000	/* number of xattrs (includes decmpfs or not depending on the -D flag)*/
#define	F_DATALESS		0x000040000000	/* dataless */
#define	F_PROTECTION_CLASS	0x000080000000	/* data protection class */
#define F_PURGEABLE		0x000100000000	/* APFS purgeable flags */
	u_int64_t	flags;			/* items set */

#define	F_BLOCK	0x001				/* block special */
#define	F_CHAR	0x002				/* char special */
#define	F_DIR	0x004				/* directory */
#define	F_FIFO	0x008				/* fifo */
#define	F_FILE	0x010				/* regular file */
#define	F_LINK	0x020				/* symbolic link */
#define	F_SOCK	0x040				/* socket */
	u_char	type;				/* file type */

	char	name[1];			/* file name (must be last) */
} NODE;

#define	RP(p)	\
	((p)->fts_path[0] == '.' && (p)->fts_path[1] == '/' ? \
	    (p)->fts_path + 2 : (p)->fts_path)

#endif /* _MTREE_H_ */
