/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
#ifndef ARCHIVE_ANDROID_LF_H_INCLUDED
#define ARCHIVE_ANDROID_LF_H_INCLUDED

#if __ANDROID_API__ > 20

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <sys/vfs.h>

//dirent.h
#define readdir_r readdir64_r
#define readdir readdir64
#define dirent dirent64
//fcntl.h
#define openat openat64
#define open open64
#define mkstemp mkstemp64
//unistd.h
#define lseek lseek64
#define ftruncate ftruncate64
//sys/stat.h
#define fstatat fstatat64
#define fstat fstat64
#define lstat lstat64
#define stat stat64
//sys/statvfs.h
#define fstatvfs fstatvfs64
#define statvfs statvfs64
//sys/types.h
#define off_t off64_t
//sys/vfs.h
#define fstatfs fstatfs64
#define statfs statfs64
#endif

#endif /* ARCHIVE_ANDROID_LF_H_INCLUDED */
