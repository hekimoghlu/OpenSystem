/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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

#include "SharedTimer.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

#if !USE(CF) && !OS(WINDOWS)
#include <wtf/RunLoop.h>
#endif

namespace WebCore {

class MainThreadSharedTimer final : public SharedTimer {
    WTF_MAKE_TZONE_ALLOCATED(MainThreadSharedTimer);
    friend class NeverDestroyed<MainThreadSharedTimer>;
public:
    static MainThreadSharedTimer& singleton();

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    void setFiredFunction(Function<void()>&&) override;
    void setFireInterval(Seconds) override;
    void stop() override;
    void invalidate() override;

    // FIXME: This should be private, but CF and Windows implementations
    // need to call this from non-member functions at the moment.
    void fired();

    WEBCORE_EXPORT static bool& shouldSetupPowerObserver();
    WEBCORE_EXPORT static void restartSharedTimer();

private:
    MainThreadSharedTimer();

    Function<void()> m_firedFunction;
#if !USE(CF) && !OS(WINDOWS)
    RunLoop::Timer m_timer;
#endif
};

} // namespace WebCore
