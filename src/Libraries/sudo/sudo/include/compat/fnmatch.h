/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#ifndef COMPAT_FNMATCH_H
#define COMPAT_FNMATCH_H

#define	FNM_NOMATCH	1		/* String does not match pattern */

#define	FNM_PATHNAME	(1 << 0)	/* Globbing chars don't match '/' */
#define	FNM_PERIOD	(1 << 1)	/* Leading '.' in string must exactly */
#define	FNM_NOESCAPE	(1 << 2)	/* Backslash treated as ordinary char */
#define	FNM_LEADING_DIR	(1 << 3)	/* Only match the leading directory */
#define	FNM_CASEFOLD	(1 << 4)	/* Case insensitive matching */

sudo_dso_public int sudo_fnmatch(const char *pattern, const char *string, int flags);

#define fnmatch(_a, _b, _c)	sudo_fnmatch((_a), (_b), (_c))

#endif /* COMPAT_FNMATCH_H */
