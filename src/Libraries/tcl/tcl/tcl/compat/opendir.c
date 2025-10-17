/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
#include "tclInt.h"

#undef DIRSIZ
#define DIRSIZ(dp) \
    ((sizeof (struct dirent) - (MAXNAMLEN+1)) + (((dp)->d_namlen+1 + 3) &~ 3))

/*
 * open a directory.
 */

DIR *
opendir(
    char *name)
{
    register DIR *dirp;
    register int fd;
    char *myname;

    myname = ((*name == '\0') ? "." : name);
    if ((fd = open(myname, 0, 0)) == -1) {
	return NULL;
    }
    dirp = (DIR *) ckalloc(sizeof(DIR));
    if (dirp == NULL) {
	/* unreachable? */
	close(fd);
	return NULL;
    }
    dirp->dd_fd = fd;
    dirp->dd_loc = 0;
    return dirp;
}

/*
 * read an old style directory entry and present it as a new one
 */
#ifndef pyr
#define	ODIRSIZ	14

struct	olddirect {
    ino_t od_ino;
    char od_name[ODIRSIZ];
};
#else	/* a Pyramid in the ATT universe */
#define	ODIRSIZ	248

struct	olddirect {
    long od_ino;
    short od_fill1, od_fill2;
    char od_name[ODIRSIZ];
};
#endif

/*
 * get next entry in a directory.
 */

struct dirent *
readdir(
    register DIR *dirp)
{
    register struct olddirect *dp;
    static struct dirent dir;

    for (;;) {
	if (dirp->dd_loc == 0) {
	    dirp->dd_size = read(dirp->dd_fd, dirp->dd_buf, DIRBLKSIZ);
	    if (dirp->dd_size <= 0) {
		return NULL;
	    }
	}
	if (dirp->dd_loc >= dirp->dd_size) {
	    dirp->dd_loc = 0;
	    continue;
	}
	dp = (struct olddirect *)(dirp->dd_buf + dirp->dd_loc);
	dirp->dd_loc += sizeof(struct olddirect);
	if (dp->od_ino == 0) {
	    continue;
	}
	dir.d_ino = dp->od_ino;
	strncpy(dir.d_name, dp->od_name, ODIRSIZ);
	dir.d_name[ODIRSIZ] = '\0'; /* insure null termination */
	dir.d_namlen = strlen(dir.d_name);
	dir.d_reclen = DIRSIZ(&dir);
	return &dir;
    }
}

/*
 * close a directory.
 */

void
closedir(
    register DIR *dirp)
{
    close(dirp->dd_fd);
    dirp->dd_fd = -1;
    dirp->dd_loc = 0;
    ckfree((char *) dirp);
}
