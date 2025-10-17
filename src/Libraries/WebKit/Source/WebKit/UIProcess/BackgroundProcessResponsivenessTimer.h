/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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

#include "ResponsivenessTimer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/RunLoop.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class WebProcessProxy;

class BackgroundProcessResponsivenessTimer : public CanMakeCheckedPtr<BackgroundProcessResponsivenessTimer> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(BackgroundProcessResponsivenessTimer);
public:
    explicit BackgroundProcessResponsivenessTimer(WebProcessProxy&);
    ~BackgroundProcessResponsivenessTimer();
    void updateState();

    void didReceiveBackgroundResponsivenessPong();
    bool isResponsive() const { return m_isResponsive; }

    void invalidate();
    void processTerminated();

private:
    Ref<WebProcessProxy> protectedWebProcessProxy() const;
    void responsivenessCheckTimerFired();
    void timeoutTimerFired();
    void setResponsive(bool);

    bool shouldBeActive() const;
    bool isActive() const;
    void scheduleNextResponsivenessCheck();
    ResponsivenessTimer::Client& client() const;

    WeakRef<WebProcessProxy> m_webProcessProxy;
    Seconds m_checkingInterval;
    RunLoop::Timer m_responsivenessCheckTimer;
    RunLoop::Timer m_timeoutTimer;
    bool m_isResponsive { true };
};

}
