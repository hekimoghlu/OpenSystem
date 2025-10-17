/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
#ifndef _DIRENT
#define _DIRENT

#ifndef _TCL
#include <tcl.h>
#endif

/*
 * Dirent structure, which holds information about a single
 * directory entry.
 */

#define MAXNAMLEN 255
#define DIRBLKSIZ 512

struct dirent {
    long d_ino;			/* Inode number of entry */
    short d_reclen;		/* Length of this record */
    short d_namlen;		/* Length of string in d_name */
    char d_name[MAXNAMLEN + 1];	/* Name must be no longer than this */
};

/*
 * State that keeps track of the reading of a directory (clients
 * should never look inside this structure;  the fields should
 * only be accessed by the library procedures).
 */

typedef struct _dirdesc {
    int dd_fd;
    long dd_loc;
    long dd_size;
    char dd_buf[DIRBLKSIZ];
} DIR;

/*
 * Procedures defined for reading directories:
 */

extern void		closedir _ANSI_ARGS_((DIR *dirp));
extern DIR *		opendir _ANSI_ARGS_((char *name));
extern struct dirent *	readdir _ANSI_ARGS_((DIR *dirp));

#endif /* _DIRENT */
