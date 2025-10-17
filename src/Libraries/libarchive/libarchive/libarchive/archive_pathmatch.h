/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
#ifndef ARCHIVE_PATHMATCH_H
#define ARCHIVE_PATHMATCH_H

#ifndef __LIBARCHIVE_BUILD
#ifndef __LIBARCHIVE_TEST
#error This header is only to be used internally to libarchive.
#endif
#endif

/* Don't anchor at beginning unless the pattern starts with "^" */
#define PATHMATCH_NO_ANCHOR_START	1
/* Don't anchor at end unless the pattern ends with "$" */
#define PATHMATCH_NO_ANCHOR_END 	2

/* Note that "^" and "$" are not special unless you set the corresponding
 * flag above. */

int __archive_pathmatch(const char *p, const char *s, int flags);
int __archive_pathmatch_w(const wchar_t *p, const wchar_t *s, int flags);

#define archive_pathmatch(p, s, f)	__archive_pathmatch(p, s, f)
#define archive_pathmatch_w(p, s, f)	__archive_pathmatch_w(p, s, f)

#endif
