/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

#include <wtf/RunLoop.h>
#include <wtf/Seconds.h>

namespace PAL {

static constexpr Seconds defaultHysteresisDuration { 5_s };

enum class HysteresisState : bool { Started, Stopped };

class HysteresisActivity {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(HysteresisActivity, PAL_EXPORT);
public:
    explicit HysteresisActivity(Function<void(HysteresisState)>&& callback = [](HysteresisState) { }, Seconds hysteresisSeconds = defaultHysteresisDuration)
        : m_callback(WTFMove(callback))
        , m_hysteresisSeconds(hysteresisSeconds)
        , m_timer(RunLoop::main(), [this] { m_callback(HysteresisState::Stopped); })
    {
    }

    void start()
    {
        if (m_active)
            return;

        m_active = true;

        if (m_timer.isActive())
            m_timer.stop();
        else
            m_callback(HysteresisState::Started);
    }

    void stop()
    {
        if (!m_active)
            return;

        m_active = false;
        m_timer.startOneShot(m_hysteresisSeconds);
    }

    void cancel()
    {
        m_active = false;
        if (m_timer.isActive())
            m_timer.stop();
    }

    void impulse()
    {
        if (m_active)
            return;

        if (state() == HysteresisState::Stopped) {
            m_active = true;
            m_callback(HysteresisState::Started);
            m_active = false;
        }

        m_timer.startOneShot(m_hysteresisSeconds);
    }

    HysteresisState state() const
    {
        return m_active || m_timer.isActive() ? HysteresisState::Started : HysteresisState::Stopped;
    }
    
private:
    Function<void(HysteresisState)> m_callback;
    Seconds m_hysteresisSeconds;
    RunLoop::Timer m_timer;
    bool m_active { false };
};

} // namespace PAL
