/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
#ifndef lint
#if 0
static char sccsid[] = "@(#)strings.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

/*
 * Mail -- a mail program
 *
 * String allocation routines.
 * Strings handed out here are reclaimed at the top of the command
 * loop each time, so they need not be freed.
 */

#include "rcv.h"
#include "extern.h"

struct strings stringdope[NSPACE];

/*
 * Allocate size more bytes of space and return the address of the
 * first byte to the caller.  An even number of bytes are always
 * allocated so that the space will always be on a word boundary.
 * The string spaces are of exponentially increasing size, to satisfy
 * the occasional user with enormous string size requests.
 */

char *
salloc(int size)
{
	char *t;
	int s, index;
	struct strings *sp;

	s = size;
	s += (sizeof(char *) - 1);
	s &= ~(sizeof(char *) - 1);
	index = 0;
	for (sp = &stringdope[0]; sp < &stringdope[NSPACE]; sp++) {
		if (sp->s_topFree == NULL && (STRINGSIZE << index) >= s)
			break;
		if (sp->s_nleft >= s)
			break;
		index++;
	}
	if (sp >= &stringdope[NSPACE])
		errx(1, "String too large");
	if (sp->s_topFree == NULL) {
		index = sp - &stringdope[0];
		if ((sp->s_topFree = malloc(STRINGSIZE << index)) == NULL)
			err(1, "No room for space %d", index);
		sp->s_nextFree = sp->s_topFree;
		sp->s_nleft = STRINGSIZE << index;
	}
	sp->s_nleft -= s;
	t = sp->s_nextFree;
	sp->s_nextFree += s;
	return (t);
}

/*
 * Reset the string area to be empty.
 * Called to free all strings allocated
 * since last reset.
 */
void
sreset(void)
{
	struct strings *sp;
	int index;

	if (noreset)
		return;
	index = 0;
	for (sp = &stringdope[0]; sp < &stringdope[NSPACE]; sp++) {
		if (sp->s_topFree == NULL)
			continue;
		sp->s_nextFree = sp->s_topFree;
		sp->s_nleft = STRINGSIZE << index;
		index++;
	}
}

/*
 * Make the string area permanent.
 * Meant to be called in main, after initialization.
 */
void
spreserve(void)
{
	struct strings *sp;

	for (sp = &stringdope[0]; sp < &stringdope[NSPACE]; sp++)
		sp->s_topFree = NULL;
}
