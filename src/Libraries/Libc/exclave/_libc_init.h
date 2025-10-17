/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
#ifndef __LIBC_INIT_H__
#define __LIBC_INIT_H__

#include <sys/cdefs.h>
#include <sys/stat.h>
#include <dirent.h>

__BEGIN_DECLS

#if defined(ENABLE_EXCLAVE_STORAGE)
struct _libc_functions {
	unsigned long version;

	// version 1
	int (*access)(const char *path, int mode);
	int (*close)(int fildes);
	ssize_t (*read)(int fildes, void *buf, size_t nbyte);
	int (*closedir)(DIR *dirp);
	DIR *(*opendir)(const char *filename);
	struct dirent *(*readdir)(DIR *dirp);
	int (*readdir_r)(DIR *dirp, struct dirent *entry, struct dirent **result);
	int (*open)(const char *path, int oflag, int mode);
	int (*fcntl)(int fildes, int cmd, void *buffer);
	int (*fstat)(int fildes, struct stat *buf);
	int (*lstat)(const char *restrict path, struct stat *restrict buf);
	int (*stat)(const char *restrict path, struct stat *restrict buf);

	// version 2
};
#else
struct _libc_functions;
#endif /* ENABLE_EXCLAVE_STORAGE */

struct ProgramVars; // forward reference

void
_libc_initializer(const struct _libc_functions *funcs,
    const char *envp[],
    const char *apple[],
    const struct ProgramVars *vars);

__END_DECLS

#endif /* __LIBC_INIT_H__ */
