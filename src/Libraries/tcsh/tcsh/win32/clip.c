/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
 * clip.c : support for clipboard functions.
 * -amol
 *
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include "sh.h"
#include "ed.h"



CCRETVAL
e_dosify_next(Char c)
{
	register Char *cp, *buf, *bp;
	int len;
    BOOL bDone = FALSE;


	USE(c);
	if (Cursor == LastChar)
		return(CC_ERROR);

	// worst case assumption
	buf = heap_alloc(( LastChar - Cursor + 1)*2*sizeof(Char));

	cp = Cursor;
	bp = buf;
	len = 0;

	while(  cp < LastChar) {
		if ( ((*cp & CHAR) == ' ') && ((cp[-1] & CHAR) != '\\') )
			bDone = TRUE;
		if (!bDone &&  (*cp & CHAR) == '/')  {
			*bp++ = '\\'  | (Char)(*cp & ~(*cp & CHAR) );
			*bp++ = '\\'  | (Char)(*cp & ~(*cp & CHAR) );

			len++;

			cp++;
		}
		else 
			*bp++ = *cp++;

		len++;
	}
	if (Cursor+ len >= InputLim) {
		heap_free(buf);
		return CC_ERROR;
	}
	cp = Cursor;
	bp = buf;
	while(len > 0) {
		*cp++ = *bp++;
		len--;
	}

	heap_free(buf);

	Cursor =  cp;

    if(LastChar < Cursor + len)
        LastChar = Cursor + len;

	return (CC_REFRESH);
}
/*ARGSUSED*/
CCRETVAL
e_dosify_prev(Char c)
{
	register Char *cp;

	USE(c);
	if (Cursor == InputBuf)
		return(CC_ERROR);
	/* else */

	cp = Cursor-1;
	/* Skip trailing spaces */
	while ((cp > InputBuf) && ( (*cp & CHAR) == ' '))
		cp--;

	while (cp > InputBuf) {
		if ( ((*cp & CHAR) == ' ') && ((cp[-1] & CHAR) != '\\') )
			break;
		cp--;
	}
	if(cp != InputBuf)
	  Cursor = cp + 1;
	else
	  Cursor = cp;
	
	return e_dosify_next(0);
}
extern BOOL ConsolePageUpOrDown(BOOL);
CCRETVAL
e_page_up(Char c) //blukas@broadcom.com
{
    USE(c);
	ConsolePageUpOrDown(TRUE);
	return (CC_REFRESH);
}
CCRETVAL
e_page_down(Char c)
{
    USE(c);
	ConsolePageUpOrDown(FALSE);
	return (CC_REFRESH);
}
