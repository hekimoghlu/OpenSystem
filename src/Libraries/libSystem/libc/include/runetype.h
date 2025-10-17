/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#ifndef	_RUNETYPE_H_
#define	_RUNETYPE_H_

#include <_types.h>

#if !defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)

#include <sys/_types/_size_t.h>
#include <sys/_types/_ct_rune_t.h>
#include <sys/_types/_rune_t.h>
#include <sys/_types/_wchar_t.h>
#include <sys/_types/_wint_t.h>

#endif /* (!_POSIX_C_SOURCE || _DARWIN_C_SOURCE) */

#define	_CACHED_RUNES	(1 <<8 )	/* Must be a power of 2 */
#define	_CRMASK		(~(_CACHED_RUNES - 1))

/*
 * The lower 8 bits of runetype[] contain the digit value of the rune.
 */
typedef struct {
	__darwin_rune_t	__min;		/* First rune of the range */
	__darwin_rune_t	__max;		/* Last rune (inclusive) of the range */
	__darwin_rune_t	__map;		/* What first maps to in maps */
	__uint32_t	*__types;	/* Array of types in range */
} _RuneEntry;

typedef struct {
	int		__nranges;	/* Number of ranges stored */
	_RuneEntry	*__ranges;	/* Pointer to the ranges */
} _RuneRange;

typedef struct {
	char		__name[14];	/* CHARCLASS_NAME_MAX = 14 */
	__uint32_t	__mask;		/* charclass mask */
} _RuneCharClass;

typedef struct {
	char		__magic[8];	/* Magic saying what version we are */
	char		__encoding[32];	/* ASCII name of this encoding */

	__darwin_rune_t	(*__sgetrune)(const char *, __darwin_size_t, char const **);
	int		(*__sputrune)(__darwin_rune_t, char *, __darwin_size_t, char **);
	__darwin_rune_t	__invalid_rune;

	__uint32_t	__runetype[_CACHED_RUNES];
	__darwin_rune_t	__maplower[_CACHED_RUNES];
	__darwin_rune_t	__mapupper[_CACHED_RUNES];

	/*
	 * The following are to deal with Runes larger than _CACHED_RUNES - 1.
	 * Their data is actually contiguous with this structure so as to make
	 * it easier to read/write from/to disk.
	 */
	_RuneRange	__runetype_ext;
	_RuneRange	__maplower_ext;
	_RuneRange	__mapupper_ext;

	void		*__variable;	/* Data which depends on the encoding */
	int		__variable_len;	/* how long that data is */

	/*
	 * extra fields to deal with arbitrary character classes
	 */
	int		__ncharclasses;
	_RuneCharClass	*__charclasses;
} _RuneLocale;

#define	_RUNE_MAGIC_A	"RuneMagA"	/* Indicates version A of RuneLocale */

__BEGIN_DECLS
extern _RuneLocale _DefaultRuneLocale;
extern _RuneLocale *_CurrentRuneLocale;
__END_DECLS

#endif	/* !_RUNETYPE_H_ */
