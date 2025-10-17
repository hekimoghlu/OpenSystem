/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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
#ifndef _LANGINFO_H_
#define	_LANGINFO_H_

#include <_bounds.h>
#include <_types.h>
#include <_types/_nl_item.h>

_LIBC_SINGLE_BY_DEFAULT()

#define	CODESET		0	/* codeset name */
#define	D_T_FMT		1	/* string for formatting date and time */
#define	D_FMT		2	/* date format string */
#define	T_FMT		3	/* time format string */
#define	T_FMT_AMPM	4	/* a.m. or p.m. time formatting string */
#define	AM_STR		5	/* Ante Meridian affix */
#define	PM_STR		6	/* Post Meridian affix */

/* week day names */
#define	DAY_1		7
#define	DAY_2		8
#define	DAY_3		9
#define	DAY_4		10
#define	DAY_5		11
#define	DAY_6		12
#define	DAY_7		13

/* abbreviated week day names */
#define	ABDAY_1		14
#define	ABDAY_2		15
#define	ABDAY_3		16
#define	ABDAY_4		17
#define	ABDAY_5		18
#define	ABDAY_6		19
#define	ABDAY_7		20

/* month names */
#define	MON_1		21
#define	MON_2		22
#define	MON_3		23
#define	MON_4		24
#define	MON_5		25
#define	MON_6		26
#define	MON_7		27
#define	MON_8		28
#define	MON_9		29
#define	MON_10		30
#define	MON_11		31
#define	MON_12		32

/* abbreviated month names */
#define	ABMON_1		33
#define	ABMON_2		34
#define	ABMON_3		35
#define	ABMON_4		36
#define	ABMON_5		37
#define	ABMON_6		38
#define	ABMON_7		39
#define	ABMON_8		40
#define	ABMON_9		41
#define	ABMON_10	42
#define	ABMON_11	43
#define	ABMON_12	44

#define	ERA		45	/* era description segments */
#define	ERA_D_FMT	46	/* era date format string */
#define	ERA_D_T_FMT	47	/* era date and time format string */
#define	ERA_T_FMT	48	/* era time format string */
#define	ALT_DIGITS	49	/* alternative symbols for digits */

#define	RADIXCHAR	50	/* radix char */
#define	THOUSEP		51	/* separator for thousands */

#define	YESEXPR		52	/* affirmative response expression */
#define	NOEXPR		53	/* negative response expression */

#if (__DARWIN_C_LEVEL > __DARWIN_C_ANSI && __DARWIN_C_LEVEL < 200112L) || __DARWIN_C_LEVEL == __DARWIN_C_FULL
#define	YESSTR		54	/* affirmative response for yes/no queries */
#define	NOSTR		55	/* negative response for yes/no queries */
#endif

#define	CRNCYSTR	56	/* currency symbol */

#if !defined(_ANSI_SOURCE) && (!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE))
#define	D_MD_ORDER	57	/* month/day order (local extension) */
#endif

__BEGIN_DECLS
char	*_LIBC_CSTR   nl_langinfo(nl_item);
__END_DECLS

#endif /* !_LANGINFO_H_ */
