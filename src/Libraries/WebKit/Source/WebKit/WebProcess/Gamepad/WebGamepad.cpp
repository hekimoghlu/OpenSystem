/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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
#include "WebGamepad.h"

#if ENABLE(GAMEPAD)

#include "GamepadData.h"
#include "Logging.h"

using WebCore::SharedGamepadValue;

namespace WebKit {

WebGamepad::WebGamepad(const GamepadData& gamepadData)
    : PlatformGamepad(gamepadData.index())
    , m_axisValues(gamepadData.axisValues().size())
    , m_buttonValues(gamepadData.buttonValues().size())
{
    LOG(Gamepad, "Connecting WebGamepad %u", gamepadData.index());

    m_id = gamepadData.id();
    m_mapping = gamepadData.mapping();
    m_supportedEffectTypes = gamepadData.supportedEffectTypes();

    updateValues(gamepadData);
}

const Vector<SharedGamepadValue>& WebGamepad::axisValues() const
{
    return m_axisValues;
}

const Vector<SharedGamepadValue>& WebGamepad::buttonValues() const
{
    return m_buttonValues;
}

void WebGamepad::updateValues(const GamepadData& gamepadData)
{
    ASSERT(gamepadData.index() == index());
    ASSERT(m_axisValues.size() == gamepadData.axisValues().size());
    ASSERT(m_buttonValues.size() == gamepadData.buttonValues().size());


    m_axisValues = WTF::map(gamepadData.axisValues(), [](auto value) { return SharedGamepadValue(value); });
    m_buttonValues = WTF::map(gamepadData.buttonValues(), [](auto value) { return SharedGamepadValue(value); });
    m_lastUpdateTime = gamepadData.lastUpdateTime();
}

}

#endif // ENABLE(GAMEPAD)
