/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#ifndef SharedTimer_h
#define SharedTimer_h

#include <wtf/Function.h>
#include <wtf/Noncopyable.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

// Each thread has its own single instance of shared timer, which implements this interface.
// This instance is shared by all timers in the thread.
// Not intended to be used directly; use the Timer class instead.
class SharedTimer {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(SharedTimer);
    WTF_MAKE_NONCOPYABLE(SharedTimer);
public:
    SharedTimer() = default;
    virtual ~SharedTimer() = default;
    virtual void setFiredFunction(Function<void()>&&) = 0;

    // The fire interval is in seconds relative to the current monotonic clock time.
    virtual void setFireInterval(Seconds) = 0;
    virtual void stop() = 0;

    virtual void invalidate() { }
};

} // namespace WebCore

#endif // SharedTimer_h
