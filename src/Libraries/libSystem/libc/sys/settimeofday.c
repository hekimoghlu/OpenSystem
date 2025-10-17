/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#include <TargetConditionals.h>
#if !TARGET_OS_DRIVERKIT
#include <notify.h>
#include <notify_keys.h>
#else
#define notify_post(...)
#endif
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "_simple.h"

#ifndef kNotifyClockSet
#define kNotifyClockSet "com.apple.system.clock_set"
#endif

int __settimeofday(const struct timeval *tp, const struct timezone *tzp);

/*
 * settimeofday stub, legacy version
 */
int
settimeofday(const struct timeval *tp, const struct timezone *tzp)
{
	int ret = __settimeofday(tp, tzp);
	if (ret == 0) notify_post(kNotifyClockSet);

	if (tp) {
		char *msg = NULL;
		asprintf(&msg, "settimeofday({%#lx,%#x}) == %d", tp->tv_sec, tp->tv_usec, ret);
		_simple_asl_log(ASL_LEVEL_NOTICE, "com.apple.settimeofday", msg);
		free(msg);
	}

	return ret;
}
