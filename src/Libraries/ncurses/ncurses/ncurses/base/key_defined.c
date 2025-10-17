/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
 *  Author: Thomas E. Dickey, 2003                                          *
 ****************************************************************************/

#include <curses.priv.h>

MODULE_ID("$Id: key_defined.c,v 1.9 2009/10/24 22:15:47 tom Exp $")

static int
find_definition(TRIES * tree, const char *str)
{
    TRIES *ptr;
    int result = OK;

    if (str != 0 && *str != '\0') {
	for (ptr = tree; ptr != 0; ptr = ptr->sibling) {
	    if (UChar(*str) == UChar(ptr->ch)) {
		if (str[1] == '\0' && ptr->child != 0) {
		    result = ERR;
		} else if ((result = find_definition(ptr->child, str + 1))
			   == OK) {
		    result = ptr->value;
		} else if (str[1] == '\0') {
		    result = ERR;
		}
	    }
	    if (result != OK)
		break;
	}
    }
    return (result);
}

/*
 * Returns the keycode associated with the given string.  If none is found,
 * return OK.  If the string is only a prefix to other strings, return ERR.
 * Otherwise, return the keycode's value (neither OK/ERR).
 */
NCURSES_EXPORT(int)
NCURSES_SP_NAME(key_defined) (NCURSES_SP_DCLx const char *str)
{
    int code = ERR;

    T((T_CALLED("key_defined(%p, %s)"), (void *) SP_PARM, _nc_visbuf(str)));
    if (SP_PARM != 0 && str != 0) {
	code = find_definition(SP_PARM->_keytry, str);
    }

    returnCode(code);
}

#if NCURSES_SP_FUNCS
NCURSES_EXPORT(int)
key_defined(const char *str)
{
    return NCURSES_SP_NAME(key_defined) (CURRENT_SCREEN, str);
}
#endif
