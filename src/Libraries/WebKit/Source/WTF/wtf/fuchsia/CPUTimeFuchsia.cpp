/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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

#include <zircon/syscalls.h>

namespace WTF {

static Seconds timeToSeconds(zx_time_t t)
{
    return Seconds(t / static_cast<double>(ZX_SEC(1)));
}

std::optional<CPUTime> CPUTime::get()
{
    // Fuchsia issue ZX-2318 tracks being able to get the monotonic and thread
    // times atomically and being able to separate ZX_CLOCK_THREAD into user and
    // kernel time.
    zx_time_t thread = 0;
    zx_clock_get(ZX_CLOCK_THREAD, &thread);

    return CPUTime { MonotonicTime::now(), timeToSeconds(thread), Seconds() };
}

Seconds CPUTime::forCurrentThread()
{
    zx_time_t thread = 0;
    zx_clock_get(ZX_CLOCK_THREAD, &thread)
    return timeToSeconds(thread);
}

}
