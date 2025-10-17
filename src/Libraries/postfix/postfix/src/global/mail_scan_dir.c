/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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
/* System library. */

#include <sys_defs.h>
#include <string.h>

/* Utility library. */

#include <scan_dir.h>

/* Global library. */

#include <mail_scan_dir.h>

/* mail_scan_dir_next - return next queue file */

char   *mail_scan_dir_next(SCAN_DIR *scan)
{
    char   *name;

    /*
     * Exploit the fact that mail queue subdirectories have one-letter names,
     * so we don't have to stat() every file in sight. This is a win because
     * many dirent implementations do not return file type information.
     */
    for (;;) {
	if ((name = scan_dir_next(scan)) == 0) {
	    if (scan_dir_pop(scan) == 0)
		return (0);
	} else if (strlen(name) == 1) {
	    scan_dir_push(scan, name);
	} else {
	    return (name);
	}
    }
}
