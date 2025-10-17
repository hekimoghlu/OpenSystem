/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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
#ifdef __APPLE__
#include <stdbool.h>
#endif

/* When to make backup files. */
enum backup_type {
	/* Never make backups. */
	none,

	/* Make simple backups of every file. */
	simple,

	/*
	 * Make numbered backups of files that already have numbered backups,
	 * and simple backups of the others.
	 */
	numbered_existing,

	/* Make numbered backups of every file. */
	numbered
};

extern enum backup_type backup_type;
extern const char	*simple_backup_suffix;
#ifdef __APPLE__
extern bool backup_mismatch;
extern bool backup_requested;
extern char	*simple_backup_prefix;
#endif

char		*find_backup_file_name(const char *file);
enum backup_type get_version(const char *version);
