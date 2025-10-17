/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#ifndef _TZLINK_H_
#define _TZLINK_H_

#include <errno.h>

/*!
 * @function tzlink
 * Create the timezone link at TZDEFAULT
 *
 * @param tz
 * New timezone, e.g. "America/Los_Angeles". This path is relative to TZDIR,
 * and must not contain any relative path components or stray slashes.
 * The file must exist and must be a valid timezone file with correct
 * ownership (root:wheel) and permissions (0644).
 *
 * @result
 * If the call succeeds, will return zero. Otherwise, returns an error:
 *   EINVAL: Invalid input, e.g. NULL or a path with relative components.
 *   ENAMETOOLONG: Input too long (generates a path > PATH_MAX)
 *   ENOENT: Specified file doesn't exist or fails owner/perm check.
 *   EPERM: Entitlement check failed.
 *   EIO: Failed to communicate with backing daemon.
 *   ENOTSUP: Always returned on OS X.
 * And possibly others not documented here.
 *
 * @discussion
 * This call can be used by any sufficiently-entitled client to overwrite
 * the timezone link at TZDEFAULT (see <tzfile.h>). It communicates with a
 * root daemon that does the necessary validation and file system work.
 * Upon success, the "SignificantTimeChangeNotification" notification is
 * posted.
 */
errno_t tzlink(const char *tz);

#endif /* !_TZLINK_H_ */
