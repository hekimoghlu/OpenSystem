/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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
#include "config.h"
#include <wtf/CPUTime.h>

#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>

namespace WTF {

static Seconds timevalToSeconds(const struct timeval& value)
{
    return Seconds(value.tv_sec) + Seconds::fromMicroseconds(value.tv_usec);
}

std::optional<CPUTime> CPUTime::get()
{
    struct rusage resource { };
    int ret = getrusage(RUSAGE_SELF, &resource);
    ASSERT_UNUSED(ret, !ret);
    return CPUTime { MonotonicTime::now(), timevalToSeconds(resource.ru_utime), timevalToSeconds(resource.ru_stime) };
}

Seconds CPUTime::forCurrentThread()
{
    struct timespec ts { };
    int ret = clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    RELEASE_ASSERT(!ret);
    return Seconds(ts.tv_sec) + Seconds::fromNanoseconds(ts.tv_nsec);
}

}
