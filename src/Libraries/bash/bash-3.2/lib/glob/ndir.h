/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#if defined (VMS)
#  if !defined (FAB$C_BID)
#    include <fab.h>
#  endif
#  if !defined (NAM$C_BID)
#    include <nam.h>
#  endif
#  if !defined (RMS$_SUC)
#    include <rmsdef.h>
#  endif
#  include "dir.h"
#endif /* VMS */

/* Size of directory block. */
#define DIRBLKSIZ 512

/* NOTE:  MAXNAMLEN must be one less than a multiple of 4 */

#if defined (VMS)
#  define MAXNAMLEN (DIR$S_NAME + 7)	/* 80 plus room for version #.  */
#  define MAXFULLSPEC NAM$C_MAXRSS	/* Maximum full spec */
#else
#  define MAXNAMLEN 15			/* Maximum filename length. */
#endif /* VMS */

/* Data from readdir (). */
struct direct {
  long d_ino;			/* Inode number of entry. */
  unsigned short d_reclen;	/* Length of this record. */
  unsigned short d_namlen;	/* Length of string in d_name. */
  char d_name[MAXNAMLEN + 1];	/* Name of file. */
};

/* Stream data from opendir (). */
typedef struct {
  int dd_fd;			/* File descriptor. */
  int dd_loc;			/* Offset in block. */
  int dd_size;			/* Amount of valid data. */
  char	dd_buf[DIRBLKSIZ];	/* Directory block. */
} DIR;

extern DIR *opendir ();
extern struct direct *readdir ();
extern long telldir ();
extern void seekdir (), closedir ();

#define rewinddir(dirp) seekdir (dirp, 0L)
