/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
 * dirent.h
 * directory interface functions. Sort of like dirent functions on unix.
 * -amol
 *
 */
#ifndef DIRENT_H
#define DIRENT_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define heap_alloc(s) HeapAlloc(GetProcessHeap(),HEAP_ZERO_MEMORY,(s))
#define heap_free(p) HeapFree(GetProcessHeap(),0,(p))
#define heap_realloc(p,s) HeapReAlloc(GetProcessHeap(),HEAP_ZERO_MEMORY,(p),(s))

#define NAME_MAX MAX_PATH

#define IS_ROOT 0x01
#define IS_NET  0x02

struct dirent {
	long            d_ino;
	int             d_off;
	unsigned short  d_reclen;
	char            d_name[NAME_MAX+1];
};

typedef struct {
	HANDLE dd_fd;
	int dd_loc;
	int dd_size;
	int flags;
	char orig_dir_name[NAME_MAX +1];
	struct dirent *dd_buf;
}DIR;

DIR *opendir(const char*);
struct dirent *readdir(DIR*);
int closedir(DIR*);
void rewinddir(DIR*);
#endif DIRENT_H
