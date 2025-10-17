/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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

#include <optional>
#include <wtf/Function.h>

namespace WebCore {

WEBCORE_EXPORT void setSystemHasBattery(bool);
WEBCORE_EXPORT bool systemHasBattery();

WEBCORE_EXPORT void resetSystemHasAC();
WEBCORE_EXPORT void setSystemHasAC(bool);
WEBCORE_EXPORT bool systemHasAC();
WEBCORE_EXPORT std::optional<bool> cachedSystemHasAC();

class WEBCORE_EXPORT SystemBatteryStatusTestingOverrides {
public:
    static SystemBatteryStatusTestingOverrides& singleton();

    void setHasAC(std::optional<bool>&&);
    std::optional<bool> hasAC() { return m_hasAC; }

    void setHasBattery(std::optional<bool>&&);
    std::optional<bool> hasBattery() { return  m_hasBattery; }

    void setConfigurationChangedCallback(std::function<void(bool)>&&);
    void resetOverridesToDefaultValues();

private:
    std::optional<bool> m_hasBattery;
    std::optional<bool> m_hasAC;
    Function<void(bool)> m_configurationChangedCallback;
};

}
