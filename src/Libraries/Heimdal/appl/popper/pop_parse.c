/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
#include <popper.h>
RCSID("$Id$");

/*
 *  parse:  Parse a raw input line from a POP client
 *  into null-delimited tokens
 */

int
pop_parse(POP *p, char *buf)
{
    char            *   mp;
    int        i;

    /*  Loop through the POP command array */
    for (mp = buf, i = 0; ; i++) {

        /*  Skip leading spaces and tabs in the message */
        while (isspace((unsigned char)*mp))mp++;

        /*  Are we at the end of the message? */
        if (*mp == 0) break;

        /*  Have we already obtained the maximum allowable parameters? */
        if (i >= MAXPARMCOUNT) {
            pop_msg(p,POP_FAILURE,"Too many arguments supplied.");
            return(-1);
        }

        /*  Point to the start of the token */
        p->pop_parm[i] = mp;

        /*  Search for the first space character (end of the token) */
        while (!isspace((unsigned char)*mp) && *mp) mp++;

        /*  Delimit the token with a null */
        if (*mp) *mp++ = 0;
    }

    /*  Were any parameters passed at all? */
    if (i == 0) return (-1);

    /*  Convert the first token (POP command) to lower case */
    strlwr(p->pop_command);

    /*  Return the number of tokens extracted minus the command itself */
    return (i-1);

}

