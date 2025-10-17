/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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

#ifndef HAVE_GMTIME_R

#include <sys/types.h>
#include <time.h>

#include "sudo_compat.h"
#include "sudo_util.h"

/*
 * Fake gmtime_r() that just stores the result.
 * Still has the normal gmtime() side effects.
 */
struct tm *
sudo_gmtime_r(const time_t *timer, struct tm *result)
{
    struct tm *tm;

    if ((tm = gmtime(timer)) == NULL)
	return NULL;
    memcpy(result, tm, sizeof(struct tm));

    return result;
}
#endif /* HAVE_GMTIME_T */
