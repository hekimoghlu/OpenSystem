/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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
 * FILE: safecalls.h
 * AUTH: Soren Spies (sspies)
 * DATE: 16 June 2006 (Copyright Apple Computer, Inc)
 * DESC: picky syscalls (constrained to one volume)
 *
 * CAVEAT: fchdir is used heavily ... until we have openat(2) and/or
 * per-thread chdir, this code is not safe to use on multiple threads.
 * we attempt to restore CWD within each call, but failure is not returned
 *
 */

#include <sys/types.h>

// secure versions of common syscalls (only if args on vol specified by fd)

// O_EXCL added if O_CREAT specified
int sopen(int fdvol, const char *path, int flags, mode_t mode);
int sopen_ifExists(int fdvol, const char *path, int flags, mode_t mode, bool failIfExists);
// WARNING: child will point to basename() [static] data
// additionally, caller must close non-(-1) olddir if requested (cf. restoredir)
int schdir(int fdvol, const char *path, int *olddir);
int schdirparent(int fdvol, const char *path, int *olddir, char childname[PATH_MAX]);
int restoredir(int savedir);        // check errors if you want them

// these are trivially implemented with the above
int smkdir(int fdvol, const char *path, mode_t mode);
int srmdir(int fdvol, const char *path);
int sunlink(int fdvol, const char *path);
// srename only renames within a directory; uses basename(newname)
int srename(int fdvol, const char *oldpath, const char *newname);

// uses FTS to recurse downwards, calling sunlink and srmdir as appropriate
int sdeepunlink(int fdvol, char *path);     // fts_open won't take const char*
// overwrite a file with zeros; attempt to ftruncate; no unlink; ENOENT okay
int szerofile(int fdvol, const char *path);
// 'mkdir -p' (recursively applies mode)
int sdeepmkdir(int fdvol, const char *path, mode_t mode);
// creates intermediate directories for you; only copies one file
int scopyitem(int srcvolfd, const char *src, int dstvolfd, const char *dst);

#ifndef STRICT_SAFETY
#define STRICT_SAFETY 1
#endif
#if STRICT_SAFETY

// #define open()               // #error use sopen (need a chicken)
#define chdir()                 // #error use schdir

#define mkdir()                 // #error use smkdir
#define rmdir()                 // #error use srmdir
#define unlink()                // #error use sunlink
#define rename()                // #error srename

#define copyfile()              // #error use scopyfile

#endif // STRICT_SAFETY
