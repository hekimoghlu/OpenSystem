/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <time.h>

#include "sudoers.h"

/*
 * Return a static buffer with the current date + time.
 */
char *
get_timestr(time_t tstamp, int log_year)
{
    static char buf[128];
    struct tm tm;
    int len;

    if (localtime_r(&tstamp, &tm) != NULL) {
	/* strftime() does not guarantee to NUL-terminate so we must check. */
	buf[sizeof(buf) - 1] = '\0';
	len = strftime(buf, sizeof(buf), log_year ? "%h %e %T %Y" : "%h %e %T",
	    &tm);
	if (len != 0 && buf[sizeof(buf) - 1] == '\0')
	    return buf;
    }
    return NULL;
}
