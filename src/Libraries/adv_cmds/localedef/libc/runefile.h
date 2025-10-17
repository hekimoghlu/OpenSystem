/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
} _FileRuneEntry;

typedef struct {
	char		magic[8];
	char		encoding[32];

	uint32_t	runetype[_CACHED_RUNES];
	int32_t		maplower[_CACHED_RUNES];
	int32_t		mapupper[_CACHED_RUNES];

	int32_t		runetype_ext_nranges;
	int32_t		maplower_ext_nranges;
	int32_t		mapupper_ext_nranges;

	int32_t		variable_len;
#ifdef __APPLE__
	int32_t		ncharclasses;
#endif
} _FileRuneLocale;

#ifdef __APPLE__
/*
 * These versions accurately portray the old format, which tried to mimic the
 * _RuneEntry/_RuneLocale structures in the on-disk format and thus, had some
 * 32-bit pointers interspersed in interesting ways.
 *
 * The future versions, above, will be the existing FreeBSD way of laying it
 * out, which just gets copied manually into a _RuneLocale rather than using
 * some more clever techniques.
 */
typedef struct {
	int32_t		min;
	int32_t		max;
	int32_t		map;
	int32_t		__types_fake;
} _FileRuneEntry_A;

typedef struct {
	char		magic[8];
	char		encoding[32];

	int32_t		__sgetrune_fake;
	int32_t		__sputrune_fake;
	int32_t		__invalid_rune;

	uint32_t	runetype[_CACHED_RUNES];
	int32_t		maplower[_CACHED_RUNES];
	int32_t		mapupper[_CACHED_RUNES];

	int32_t		runetype_ext_nranges;
	int32_t		__runetype_ext_ranges_fake;
	int32_t		maplower_ext_nranges;
	int32_t		__maplower_ext_ranges_fake;
	int32_t		mapupper_ext_nranges;
	int32_t		__mapupper_ext_ranges_fake;

	int32_t		__variable_fake;
	int32_t		variable_len;

	int32_t		ncharclasses;
	int32_t		__charclasses_fake;
} _FileRuneLocale_A;

typedef struct {
	char		name[14];	/* CHARCLASS_NAME_MAX = 14 */
	__uint32_t	mask;		/* charclass mask */
} _FileRuneCharClass;

#define	_FILE_RUNE_MAGIC_A	"RuneMagA"	/* Indicates version A of RuneLocale */
#define	_FILE_RUNE_MAGIC_B	"RuneMagB"	/* Indicates version B of RuneLocale */
#endif
#define	_FILE_RUNE_MAGIC_1	"RuneMag1"

#endif	/* !_RUNEFILE_H_ */
