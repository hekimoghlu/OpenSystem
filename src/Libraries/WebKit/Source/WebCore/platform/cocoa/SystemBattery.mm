/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
#import "config.h"
#import "SystemBattery.h"

#import <notify.h>
#import <pal/spi/cocoa/IOPSLibSPI.h>

namespace WebCore {

static std::optional<bool> hasBattery;
static std::optional<bool> hasAC;

void setSystemHasBattery(bool battery)
{
    hasBattery = battery;
}

bool systemHasBattery()
{
    if (auto overrideForTesting = SystemBatteryStatusTestingOverrides::singleton().hasBattery())
        return *overrideForTesting;

    if (!hasBattery.has_value()) {
        hasBattery = [] {
#if PLATFORM(IOS) || PLATFORM(WATCHOS) || PLATFORM(VISION)
            // Devices running iOS / WatchOS always have a battery.
            return true;
#elif PLATFORM(APPLETV)
            return false;
#else
            RetainPtr<CFTypeRef> powerSourcesInfo = adoptCF(IOPSCopyPowerSourcesInfo());
            if (!powerSourcesInfo)
                return false;
            RetainPtr<CFArrayRef> powerSourcesList = adoptCF(IOPSCopyPowerSourcesList(powerSourcesInfo.get()));
            if (!powerSourcesList)
                return false;
            for (CFIndex i = 0, count = CFArrayGetCount(powerSourcesList.get()); i < count; ++i) {
                CFDictionaryRef description = IOPSGetPowerSourceDescription(powerSourcesInfo.get(), CFArrayGetValueAtIndex(powerSourcesList.get(), i));
                CFTypeRef value = CFDictionaryGetValue(description, CFSTR(kIOPSTypeKey));
                if (!value || CFEqual(value, CFSTR(kIOPSInternalBatteryType)))
                    return true;
            }
            return false;
#endif
        }();
    }

    return *hasBattery;
}

void resetSystemHasAC()
{
    hasAC.reset();
}

void setSystemHasAC(bool ac)
{
    hasAC = ac;
}

bool systemHasAC()
{
    if (auto overrideForTesting = SystemBatteryStatusTestingOverrides::singleton().hasAC())
        return *overrideForTesting;

    if (!hasAC.has_value()) {
        hasAC = [] {
#if PLATFORM(APPLETV)
            return true;
#else
            RetainPtr<CFTypeRef> powerSourcesInfo = adoptCF(IOPSCopyPowerSourcesInfo());
            if (!powerSourcesInfo)
                return false;
            RetainPtr<CFArrayRef> powerSourcesList = adoptCF(IOPSCopyPowerSourcesList(powerSourcesInfo.get()));
            if (!powerSourcesList)
                return false;
            for (CFIndex i = 0, count = CFArrayGetCount(powerSourcesList.get()); i < count; ++i) {
                CFDictionaryRef description = IOPSGetPowerSourceDescription(powerSourcesInfo.get(), CFArrayGetValueAtIndex(powerSourcesList.get(), i));
                if (!description)
                    continue;
                CFTypeRef value = CFDictionaryGetValue(description, CFSTR(kIOPSPowerSourceStateKey));
                if (value && CFEqual(value, CFSTR(kIOPSACPowerValue)))
                    return true;
            }
            return false;
#endif
        }();
    }

    return *hasAC;
}

std::optional<bool> cachedSystemHasAC()
{
    return hasAC;
}

SystemBatteryStatusTestingOverrides& SystemBatteryStatusTestingOverrides::singleton()
{
    static NeverDestroyed<SystemBatteryStatusTestingOverrides> instance;
    return instance;
}

void SystemBatteryStatusTestingOverrides::setHasBattery(std::optional<bool>&& hasBattery)
{
    m_hasBattery = WTFMove(hasBattery);
    if (m_configurationChangedCallback)
        m_configurationChangedCallback(false);
}

void SystemBatteryStatusTestingOverrides::setHasAC(std::optional<bool>&& hasAC)
{
    m_hasAC = WTFMove(hasAC);
    if (m_configurationChangedCallback)
        m_configurationChangedCallback(false);
}

void SystemBatteryStatusTestingOverrides::setConfigurationChangedCallback(std::function<void(bool)>&& callback)
{
    m_configurationChangedCallback = WTFMove(callback);
}

void SystemBatteryStatusTestingOverrides::resetOverridesToDefaultValues()
{
    setHasBattery(std::nullopt);
    setHasAC(std::nullopt);
    if (m_configurationChangedCallback)
        m_configurationChangedCallback(true);
}

}
