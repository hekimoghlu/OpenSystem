/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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

#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

#if HAVE(APPLE_THERMAL_MITIGATION_SUPPORT)
#include <wtf/RetainPtr.h>
OBJC_CLASS WebThermalMitigationObserver;
#endif

namespace WebCore {
class ThermalMitigationNotifier;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::ThermalMitigationNotifier> : std::true_type { };
}

namespace WebCore {

class ThermalMitigationNotifier : public CanMakeWeakPtr<ThermalMitigationNotifier> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ThermalMitigationNotifier, WEBCORE_EXPORT);
public:
    using ThermalMitigationChangeCallback = Function<void(bool thermalMitigationEnabled)>;
    WEBCORE_EXPORT explicit ThermalMitigationNotifier(ThermalMitigationChangeCallback&&);
    WEBCORE_EXPORT ~ThermalMitigationNotifier();

    WEBCORE_EXPORT bool thermalMitigationEnabled() const;
    WEBCORE_EXPORT static bool isThermalMitigationEnabled();

private:
#if HAVE(APPLE_THERMAL_MITIGATION_SUPPORT)
    void notifyThermalMitigationChanged(bool);
    friend void notifyThermalMitigationChanged(ThermalMitigationNotifier&, bool);

    RetainPtr<WebThermalMitigationObserver> m_observer;
    ThermalMitigationChangeCallback m_callback;
#endif
};

}
