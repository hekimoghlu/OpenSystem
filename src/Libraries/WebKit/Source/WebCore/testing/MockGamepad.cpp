/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
#include "config.h"
#include "MockGamepad.h"

#if ENABLE(GAMEPAD)

namespace WebCore {

MockGamepad::MockGamepad(unsigned index, const String& gamepadID, const String& mapping, unsigned axisCount, unsigned buttonCount, bool supportsDualRumble)
    : PlatformGamepad(index)
{
    m_connectTime = m_lastUpdateTime = MonotonicTime::now();
    updateDetails(gamepadID, mapping, axisCount, buttonCount, supportsDualRumble);
}

void MockGamepad::updateDetails(const String& gamepadID, const String& mapping, unsigned axisCount, unsigned buttonCount, bool supportsDualRumble)
{
    m_id = gamepadID;
    m_mapping = mapping;
    m_axisValues.clear();
    for (size_t i = 0; i < axisCount; ++i)
        m_axisValues.append({ });
    m_buttonValues.clear();
    for (size_t i = 0; i < buttonCount; ++i)
        m_buttonValues.append({ });
    m_lastUpdateTime = MonotonicTime::now();
    if (supportsDualRumble)
        m_supportedEffectTypes.add(GamepadHapticEffectType::DualRumble);
    else
        m_supportedEffectTypes.remove(GamepadHapticEffectType::DualRumble);
}

bool MockGamepad::setAxisValue(unsigned index, double value)
{
    if (index >= m_axisValues.size()) {
        LOG_ERROR("MockGamepad (%u): Attempt to set value on axis %u which doesn't exist", m_index, index);
        return false;
    }

    m_axisValues[index].setValue(value);
    m_lastUpdateTime = MonotonicTime::now();
    return true;
}

bool MockGamepad::setButtonValue(unsigned index, double value)
{
    if (index >= m_buttonValues.size()) {
        LOG_ERROR("MockGamepad (%u): Attempt to set value on button %u which doesn't exist", m_index, index);
        return false;
    }

    m_buttonValues[index].setValue(value);
    m_lastUpdateTime = MonotonicTime::now();
    return true;
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
