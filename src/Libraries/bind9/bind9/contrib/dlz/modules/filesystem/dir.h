/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
#include <sys/types.h>
#include <dirent.h>

#define DIR_NAMEMAX 256
#define DIR_PATHMAX 1024

typedef struct direntry {
	char 		name[DIR_NAMEMAX];
	unsigned int	length;
} direntry_t;

typedef struct dir {
	char		dirname[DIR_PATHMAX];
	direntry_t	entry;
	DIR *		handle;
} dir_t;

void
dir_init(dir_t *dir);

isc_result_t
dir_open(dir_t *dir, const char *dirname);

isc_result_t
dir_read(dir_t *dir);

isc_result_t
dir_reset(dir_t *dir);

void
dir_close(dir_t *dir);
