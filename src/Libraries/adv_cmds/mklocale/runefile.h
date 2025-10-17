/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
#ifndef _RUNEFILE_H_
#define	_RUNEFILE_H_

#include <sys/types.h>

#ifndef _CACHED_RUNES
#define	_CACHED_RUNES	(1 << 8)
#endif

typedef struct {
	int32_t		min;
	int32_t		max;
	int32_t		map;
#ifdef __APPLE__
	int32_t		__types_fake;
#endif /* __APPLE__ */
} _FileRuneEntry;

typedef struct {
	char		magic[8];
	char		encoding[32];

#ifdef __APPLE__
	int32_t		__sgetrune_fake;
	int32_t		__sputrune_fake;
	int32_t		__invalid_rune;
#endif /* __APPLE__ */

	uint32_t	runetype[_CACHED_RUNES];
	int32_t		maplower[_CACHED_RUNES];
	int32_t		mapupper[_CACHED_RUNES];

	int32_t		runetype_ext_nranges;
#ifdef __APPLE__
	int32_t		__runetype_ext_ranges_fake;
#endif /* __APPLE__ */
	int32_t		maplower_ext_nranges;
#ifdef __APPLE__
	int32_t		__maplower_ext_ranges_fake;
#endif /* __APPLE__ */
	int32_t		mapupper_ext_nranges;
#ifdef __APPLE__
	int32_t		__mapupper_ext_ranges_fake;
#endif /* __APPLE__ */

#ifdef __APPLE__
	int32_t		__variable_fake;
#endif /* __APPLE__ */
	int32_t		variable_len;

#ifdef __APPLE__
	int32_t		ncharclasses;
	int32_t		__charclasses_fake;
#endif /* __APPLE__ */
} _FileRuneLocale;

#define	_FILE_RUNE_MAGIC_1	"RuneMag1"

#endif	/* !_RUNEFILE_H_ */
