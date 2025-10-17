/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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

#include "Clock.h"
#include <wtf/MonotonicTime.h>

namespace PAL {

class ClockGeneric final : public Clock {
    WTF_MAKE_TZONE_ALLOCATED(ClockGeneric);
public:
    ClockGeneric();

private:
    void setCurrentTime(double) override;
    double currentTime() const override;

    void setPlayRate(double) override;
    double playRate() const override { return m_rate; }

    void start() override;
    void stop() override;
    bool isRunning() const override { return m_running; }

    MonotonicTime now() const;
    double currentDelta() const;

    bool m_running;
    double m_rate;
    double m_offset;
    MonotonicTime m_startTime;
    mutable MonotonicTime m_lastTime;
};

}
