/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
 * Routines to perform bracket matching functions.
 */

#include "less.h"
#include "position.h"

/*
 * Try to match the n-th open bracket 
 *  which appears in the top displayed line (forwdir),
 * or the n-th close bracket 
 *  which appears in the bottom displayed line (!forwdir).
 * The characters which serve as "open bracket" and 
 * "close bracket" are given.
 */
	public void
match_brac(obrac, cbrac, forwdir, n)
	int obrac;
	int cbrac;
	int forwdir;
	int n;
{
	int c;
	int nest;
	POSITION pos;
	int (*chget)();

	extern int ch_forw_get(), ch_back_get();

	/*
	 * Seek to the line containing the open bracket.
	 * This is either the top or bottom line on the screen,
	 * depending on the type of bracket.
	 */
	pos = position((forwdir) ? TOP : BOTTOM);
	if (pos == NULL_POSITION || ch_seek(pos))
	{
		if (forwdir)
			error("Nothing in top line", NULL_PARG);
		else
			error("Nothing in bottom line", NULL_PARG);
		return;
	}

	/*
	 * Look thru the line to find the open bracket to match.
	 */
	do
	{
		if ((c = ch_forw_get()) == '\n' || c == EOI)
		{
			if (forwdir)
				error("No bracket in top line", NULL_PARG);
			else
				error("No bracket in bottom line", NULL_PARG);
			return;
		}
	} while (c != obrac || --n > 0);

	/*
	 * Position the file just "after" the open bracket
	 * (in the direction in which we will be searching).
	 * If searching forward, we are already after the bracket.
	 * If searching backward, skip back over the open bracket.
	 */
	if (!forwdir)
		(void) ch_back_get();

	/*
	 * Search the file for the matching bracket.
	 */
	chget = (forwdir) ? ch_forw_get : ch_back_get;
	nest = 0;
	while ((c = (*chget)()) != EOI)
	{
		if (c == obrac)
			nest++;
		else if (c == cbrac && --nest < 0)
		{
			/*
			 * Found the matching bracket.
			 * If searching backward, put it on the top line.
			 * If searching forward, put it on the bottom line.
			 */
			jump_line_loc(ch_tell(), forwdir ? -1 : 1);
			return;
		}
	}
	error("No matching bracket", NULL_PARG);
}
