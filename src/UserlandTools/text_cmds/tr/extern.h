/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#include <limits.h>

#define	NCHARS_SB	(UCHAR_MAX + 1)	/* Number of single-byte characters. */
#define	OOBCH		-1		/* Out of band character value. */

typedef struct {
	enum { STRING1, STRING2 } which;
	enum { EOS, INFINITE, NORMAL, RANGE, SEQUENCE,
	       CCLASS, CCLASS_UPPER, CCLASS_LOWER, SET } state;
	int		cnt;		/* character count */
	wint_t		lastch;		/* last character */
	wctype_t	cclass;		/* character class from wctype() */
	wint_t		equiv[NCHARS_SB];	/* equivalence set */
	wint_t		*set;		/* set of characters */
	char		*str;		/* user's string */
} STR;

wint_t	 next(STR *);
int charcoll(const void *, const void *);
