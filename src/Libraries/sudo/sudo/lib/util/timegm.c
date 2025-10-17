/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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

#ifndef HAVE_TIMEGM

#include <stdio.h>
#include <time.h>

#include "sudo_compat.h"
#include "sudo_debug.h"

/*
 * Returns the offset from GMT in seconds (algorithm taken from sendmail).
 */
#ifdef HAVE_STRUCT_TM_TM_GMTOFF
static long
get_gmtoff(time_t *when)
{
    struct tm local;

    if (localtime_r(when, &local) == NULL)
	return 0;

    /* Adjust for DST. */
    if (local.tm_isdst != 0)
	local.tm_gmtoff -= local.tm_isdst * 3600;

    return local.tm_gmtoff;
}
#else
static long
get_gmtoff(time_t *when)
{
    struct tm gmt, local;
    long offset;

    if (gmtime_r(when, &gmt) == NULL)
	return 0;
    if (localtime_r(when, &local) == NULL)
	return 0;

    offset = (local.tm_sec - gmt.tm_sec) +
	((local.tm_min - gmt.tm_min) * 60) +
	((local.tm_hour - gmt.tm_hour) * 3600);

    /* Timezone may cause year rollover to happen on a different day. */
    if (local.tm_year < gmt.tm_year)
	offset -= 24 * 3600;
    else if (local.tm_year > gmt.tm_year)
	offset -= 24 * 3600;
    else if (local.tm_yday < gmt.tm_yday)
	offset -= 24 * 3600;
    else if (local.tm_yday > gmt.tm_yday)
	offset += 24 * 3600;

    /* Adjust for DST. */
    if (local.tm_isdst != 0)
	offset -= local.tm_isdst * 3600;

    return offset;
}
#endif /* HAVE_TM_GMTOFF */

time_t
sudo_timegm(struct tm *tm)
{
    time_t result;

    tm->tm_isdst = 0;
    result = mktime(tm);
    if (result != -1)
	result += get_gmtoff(&result);

    return result;
}

#endif /* HAVE_TIMEGM */
