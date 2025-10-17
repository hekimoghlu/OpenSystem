/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#include "heim.h"
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

mit_krb5_error_code KRB5_CALLCONV
krb5_string_to_timestamp(char *string, mit_krb5_timestamp *timestampp)
{
    int i;
    struct tm timebuf, timebuf2;
    time_t now, ret_time;
    char *s;
    static const char * const atime_format_table[] = {
	"%Y%m%d%H%M%S",		/* yyyymmddhhmmss		*/
	"%Y.%m.%d.%H.%M.%S",	/* yyyy.mm.dd.hh.mm.ss		*/
	"%y%m%d%H%M%S",		/* yymmddhhmmss			*/
	"%y.%m.%d.%H.%M.%S",	/* yy.mm.dd.hh.mm.ss		*/
	"%y%m%d%H%M",		/* yymmddhhmm			*/
	"%H%M%S",		/* hhmmss			*/
	"%H%M",			/* hhmm				*/
	"%T",			/* hh:mm:ss			*/
	"%R",			/* hh:mm			*/
	/* The following not really supported unless native strptime present */
	"%x:%X",		/* locale-dependent short format */
	"%d-%b-%Y:%T",		/* dd-month-yyyy:hh:mm:ss	*/
	"%d-%b-%Y:%R"		/* dd-month-yyyy:hh:mm		*/
    };
    static const int atime_format_table_nents =
	sizeof(atime_format_table)/sizeof(atime_format_table[0]);


    now = time((time_t *) NULL);
    if (localtime_r(&now, &timebuf2) == NULL)
	return EINVAL;
    for (i=0; i<atime_format_table_nents; i++) {
        /* We reset every time throughout the loop as the manual page
	 * indicated that no guarantees are made as to preserving timebuf
	 * when parsing fails
	 */
	timebuf = timebuf2;
	if ((s = strptime(string, atime_format_table[i], &timebuf))
	    && (s != string)) {
 	    /* See if at end of buffer - otherwise partial processing */
	    while(*s != 0 && isspace((unsigned char) *s)) s++;
	    if (*s != 0)
	        continue;
	    if (timebuf.tm_year <= 0)
		continue;	/* clearly confused */
	    ret_time = mktime(&timebuf);
	    if (ret_time == (time_t) -1)
		continue;	/* clearly confused */
	    *timestampp = (mit_krb5_timestamp) ret_time;
	    return 0;
	}
    }
    return(EINVAL);
}
