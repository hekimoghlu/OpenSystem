/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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
	userdefines.h

	Header file that contains the major user-defineable quantities for the Counterpane PRNG.
*/
#ifndef __YARROW_USER_DEFINES_H__
#define __YARROW_USER_DEFINES_H__

/* User-alterable define statements */
#define STRICT				/* Define to force strict type checking */
#define K 0					/* How many sources should we ignore when calculating total entropy? */
#define THRESHOLD 100		/* Minimum amount of entropy for a reseed */
#define BACKTRACKLIMIT 500	/* Number of outputed bytes after which to generate a new state */
#define COMPRESSION_ON		/* Define this variable to add on-the-fly compression (recommended) */
							/* for user sources */
#if		!defined(macintosh) && !defined(__APPLE__)
#define WIN_95				/* Choose an OS: WIN_95, WIN_NT */
#endif

/* Setup Microsoft flag for NT4.0 */
#ifdef WIN_NT
#define _WIN32_WINNT 0x0400
#endif

#endif	/* __YARROW_USER_DEFINES_H__ */
