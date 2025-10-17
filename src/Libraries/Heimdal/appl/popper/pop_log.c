/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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
 *  log:    Make a log entry
 */

int
pop_log(POP *p, int stat, char *format, ...)
{
    char msgbuf[MAXLINELEN];
    va_list     ap;

    va_start(ap, format);
    vsnprintf(msgbuf, sizeof(msgbuf), format, ap);

    if (p->debug && p->trace) {
        fprintf(p->trace,"%s\n",msgbuf);
        fflush(p->trace);
    } else {
#ifdef KRB5
	krb5_log(p->context, p->logf, stat, "%s", msgbuf);
#else
        syslog (stat,"%s",msgbuf);
#endif
    }
    va_end(ap);

    return(stat);
}
