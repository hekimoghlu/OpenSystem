/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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
#pragma once

#include <algorithm>
#include <wtf/ApproximateTime.h>
#include <wtf/TimeWithDynamicClockType.h>

namespace IPC {

class Timeout {
public:
    Timeout(Seconds timeDelta)
        : m_deadline(timeDelta.isInfinity() ? ApproximateTime::infinity() : ApproximateTime::now() + timeDelta)
    {
    }

    static constexpr Timeout infinity() { return Timeout { }; }
    bool isInfinity() const { return m_deadline.isInfinity(); }
    static Timeout now() { return 0_s; }
    Seconds secondsUntilDeadline() const { return std::max(m_deadline - ApproximateTime::now(), 0_s ); }
    constexpr ApproximateTime deadline() const { return m_deadline; }
    bool didTimeOut() const { return ApproximateTime::now() >= m_deadline; }

private:
    explicit constexpr Timeout()
        : m_deadline(ApproximateTime::infinity())
    {
    }

    ApproximateTime m_deadline;
};

}
