/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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
/****************************************************************************
 *  Author: Thomas E. Dickey            1998-on                             *
 ****************************************************************************/

/*
**	add_tries.c
**
**	Add keycode/string to tries-tree.
**
*/

#include <curses.priv.h>

MODULE_ID("$Id: add_tries.c,v 1.10 2010/12/19 01:31:14 tom Exp $")

#define SET_TRY(dst,src) if ((dst->ch = *src++) == 128) dst->ch = '\0'
#define CMP_TRY(a,b) ((a)? (a == b) : (b == 128))

NCURSES_EXPORT(int)
_nc_add_to_try(TRIES ** tree, const char *str, unsigned code)
{
    TRIES *ptr, *savedptr;
    unsigned const char *txt = (unsigned const char *) str;

    T((T_CALLED("_nc_add_to_try(%p, %s, %u)"),
       (void *) *tree, _nc_visbuf(str), code));
    if (txt == 0 || *txt == '\0' || code == 0)
	returnCode(ERR);

    if ((*tree) != 0) {
	ptr = savedptr = (*tree);

	for (;;) {
	    unsigned char cmp = *txt;

	    while (!CMP_TRY(ptr->ch, cmp)
		   && ptr->sibling != 0)
		ptr = ptr->sibling;

	    if (CMP_TRY(ptr->ch, cmp)) {
		if (*(++txt) == '\0') {
		    ptr->value = (unsigned short) code;
		    returnCode(OK);
		}
		if (ptr->child != 0)
		    ptr = ptr->child;
		else
		    break;
	    } else {
		if ((ptr->sibling = typeCalloc(TRIES, 1)) == 0) {
		    returnCode(ERR);
		}

		savedptr = ptr = ptr->sibling;
		SET_TRY(ptr, txt);
		ptr->value = 0;

		break;
	    }
	}			/* end for (;;) */
    } else {			/* (*tree) == 0 :: First sequence to be added */
	savedptr = ptr = (*tree) = typeCalloc(TRIES, 1);

	if (ptr == 0) {
	    returnCode(ERR);
	}

	SET_TRY(ptr, txt);
	ptr->value = 0;
    }

    /* at this point, we are adding to the try.  ptr->child == 0 */

    while (*txt) {
	ptr->child = typeCalloc(TRIES, 1);

	ptr = ptr->child;

	if (ptr == 0) {
	    while ((ptr = savedptr) != 0) {
		savedptr = ptr->child;
		free(ptr);
	    }
	    returnCode(ERR);
	}

	SET_TRY(ptr, txt);
	ptr->value = 0;
    }

    ptr->value = (unsigned short) code;
    returnCode(OK);
}
