/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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
#if 0
static char rcsid[] = "$FreeBSD: /repoman/r/ncvs/src/lib/libcompat/4.1/ftime.c,v 1.5 1999/08/28 00:04:12 peter Exp $";
#endif /* not lint */

#include <sys/types.h>
#include <sys/time.h>
#include <sys/timeb.h>

int
ftime(struct timeb *tbp)
{
        struct timezone tz;
        struct timeval t;

        if (gettimeofday(&t, &tz) < 0)
                return (-1);
        tbp->millitm = t.tv_usec / 1000;
        tbp->time = t.tv_sec;
        tbp->timezone = tz.tz_minuteswest;
        tbp->dstflag = tz.tz_dsttime;

	return (0);
}
