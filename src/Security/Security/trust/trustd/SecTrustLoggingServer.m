/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
#include <AssertMacros.h>
#include "SecTrustLoggingServer.h"


uint64_t TimeSinceSystemStartup(void) {
    struct timespec uptime;
    clock_gettime(CLOCK_UPTIME_RAW, &uptime);
    return (uint64_t)uptime.tv_sec;
}

uint64_t TimeSinceProcessLaunch(void) {
    return mach_absolute_time() - launchTime;
}

int64_t TimeUntilProcessUptime(int64_t uptime_nsecs) {
    int64_t uptime = (int64_t)TimeSinceProcessLaunch();
    if (uptime > 0 && uptime < uptime_nsecs) {
        return uptime_nsecs - uptime;
    }
    return 0;
}
