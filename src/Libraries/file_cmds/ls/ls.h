/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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
#ifndef _LS_H_
#define _LS_H_

#define NO_PRINT	1

#define HUMANVALSTR_LEN	5

extern long blocksize;		/* block size units */

extern int f_accesstime;	/* use time of last access */
extern int f_birthtime;	/* use time of file creation */
extern int f_flags;		/* show flags associated with a file */
extern int f_humanval;		/* show human-readable file sizes */
#ifndef __APPLE__
extern int f_label;		/* show MAC label */
#endif
extern int f_inode;		/* print inode */
extern int f_longform;		/* long listing format */
extern int f_octal;		/* print unprintables in octal */
extern int f_octal_escape;	/* like f_octal but use C escapes if possible */
extern int f_nonprint;		/* show unprintables as ? */
extern int f_samesort;		/* sort time and name in same direction */
extern int f_sectime;		/* print the real time for all files */
extern int f_size;		/* list size in short listing */
extern int f_slash;		/* append a '/' if the file is a directory */
extern int f_sortacross;	/* sort across rows, not down columns */
extern int f_statustime;	/* use time of last mode change */
extern int f_thousands;		/* show file sizes with thousands separators */
extern char *f_timeformat;	/* user-specified time format */
extern int f_notabs;		/* don't use tab-separated multi-col output */
extern int f_type;		/* add type character for non-regular files */
#ifdef __APPLE__
extern int f_acl;		/* print ACLs in long format */
extern int f_xattr;		/* print extended attributes in long format  */
extern int f_group;		/* list group without owner */
extern int f_owner;		/* list owner without group */
#endif
#ifdef COLORLS
extern int f_color;		/* add type in color for non-regular files */
#endif
extern int f_numericonly;	/* don't convert uid/gid to name */

#ifdef __APPLE__
#include <sys/acl.h>
#endif // __APPLE__

typedef struct {
	FTSENT *list;
	u_int64_t btotal;
	int entries;
	int maxlen;
	u_int s_block;
	u_int s_flags;
#ifndef __APPLE__
	u_int s_label;
#endif
	u_int s_group;
	u_int s_inode;
	u_int s_nlink;
	u_int s_size;
	u_int s_user;
} DISPLAY;

typedef struct {
	char *user;
	char *group;
	char *flags;
#ifndef __APPLE__
	char *label;
#else
	char	*xattr_names;	/* f_xattr */
	int	*xattr_sizes;
	acl_t	acl;		/* f_acl */
        int	xattr_count;
	char	mode_suffix;	/* @ | + | % | <space> */
#endif /* __APPLE__ */
	char data[1];
} NAMES;

#endif /* _LS_H_ */
