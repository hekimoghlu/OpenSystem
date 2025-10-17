/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
 *  msg:    Send a formatted line to the POP client
 */

int
pop_msg(POP *p, int stat, const char *format, ...)
{
    char	       *mp;
    char                message[MAXLINELEN];
    va_list             ap;

    va_start(ap, format);

    /*  Point to the message buffer */
    mp = message;

    /*  Format the POP status code at the beginning of the message */
    snprintf (mp, sizeof(message), "%s ",
	      (stat == POP_SUCCESS) ? POP_OK : POP_ERR);

    /*  Point past the POP status indicator in the message message */
    mp += strlen(mp);

    /*  Append the message (formatted, if necessary) */
    if (format)
	vsnprintf (mp, sizeof(message) - strlen(message),
		   format, ap);

    /*  Log the message if debugging is turned on */
#ifdef DEBUG
    if (p->debug && stat == POP_SUCCESS)
        pop_log(p,POP_DEBUG,"%s",message);
#endif /* DEBUG */

    /*  Log the message if a failure occurred */
    if (stat != POP_SUCCESS)
        pop_log(p,POP_PRIORITY,"%s",message);

    /*  Append the <CR><LF> */
    strlcat(message, "\r\n", sizeof(message));

    /*  Send the message to the client */
    fputs(message, p->output);
    fflush(p->output);

    va_end(ap);
    return(stat);
}
