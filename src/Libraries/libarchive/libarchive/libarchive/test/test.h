/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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
/* Every test program should #include "test.h" as the first thing. */

#define KNOWNREF	"test_compat_gtar_1.tar.uu"
#define	ENVBASE "LIBARCHIVE" /* Prefix for environment variables. */
#undef	PROGRAM              /* Testing a library, not a program. */
#define	LIBRARY	"libarchive"
#define	EXTRA_DUMP(x)	archive_error_string((struct archive *)(x))
#define	EXTRA_ERRNO(x)	archive_errno((struct archive *)(x))
#define	EXTRA_VERSION	archive_version_details()

#if defined(__GNUC__) && (__GNUC__ >= 7)
#define	__LA_FALLTHROUGH	__attribute__((fallthrough))
#else
#define	__LA_FALLTHROUGH
#endif

#include "test_common.h"
