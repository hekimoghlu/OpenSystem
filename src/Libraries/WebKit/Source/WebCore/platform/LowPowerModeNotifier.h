/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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

#include <wtf/CheckedPtr.h>
#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>

#if HAVE(APPLE_LOW_POWER_MODE_SUPPORT)
#include <wtf/RetainPtr.h>
OBJC_CLASS WebLowPowerModeObserver;
#endif

#if USE(GLIB)
#include <wtf/glib/GRefPtr.h>
extern "C" {
typedef struct _GPowerProfileMonitor GPowerProfileMonitor;
};
#endif

namespace WebCore {

class LowPowerModeNotifier : public CanMakeCheckedPtr<LowPowerModeNotifier> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(LowPowerModeNotifier, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LowPowerModeNotifier);
public:
    using LowPowerModeChangeCallback = Function<void(bool isLowPowerModeEnabled)>;
    WEBCORE_EXPORT explicit LowPowerModeNotifier(LowPowerModeChangeCallback&&);
    WEBCORE_EXPORT ~LowPowerModeNotifier();

    WEBCORE_EXPORT bool isLowPowerModeEnabled() const;

private:
#if HAVE(APPLE_LOW_POWER_MODE_SUPPORT)
    void notifyLowPowerModeChanged(bool);
    friend void notifyLowPowerModeChanged(LowPowerModeNotifier&, bool);

    RetainPtr<WebLowPowerModeObserver> m_observer;
    LowPowerModeChangeCallback m_callback;
#elif USE(GLIB)
    LowPowerModeChangeCallback m_callback;
    GRefPtr<GPowerProfileMonitor> m_powerProfileMonitor;
    bool m_lowPowerModeEnabled { false };
#endif
};

}
