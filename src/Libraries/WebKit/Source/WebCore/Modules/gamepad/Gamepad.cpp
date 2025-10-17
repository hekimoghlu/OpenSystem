/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#include "Gamepad.h"

#if ENABLE(GAMEPAD)

#include "GamepadButton.h"
#include "GamepadHapticActuator.h"
#include "PlatformGamepad.h"
#include "ScriptExecutionContext.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

Gamepad::Gamepad(Document* document, const PlatformGamepad& platformGamepad)
    : m_id(platformGamepad.id())
    , m_index(platformGamepad.index())
    , m_connected(true)
    , m_timestamp(platformGamepad.lastUpdateTime())
    , m_mapping(platformGamepad.mapping())
    , m_supportedEffectTypes(platformGamepad.supportedEffectTypes())
    , m_axes(platformGamepad.axisValues().size(), 0.0)
    , m_vibrationActuator(platformGamepad.supportedEffectTypes().contains(GamepadHapticEffectType::DualRumble) ? RefPtr { GamepadHapticActuator::create(document, GamepadHapticActuator::Type::DualRumble, *this) } : nullptr)
{
    unsigned buttonCount = platformGamepad.buttonValues().size();
    m_buttons = Vector<Ref<GamepadButton>>(buttonCount, [](size_t) {
        return GamepadButton::create();
    });
}

Gamepad::~Gamepad() = default;

const Vector<double>& Gamepad::axes() const
{
    return m_axes;
}

const Vector<Ref<GamepadButton>>& Gamepad::buttons() const
{
    return m_buttons;
}

void Gamepad::updateFromPlatformGamepad(const PlatformGamepad& platformGamepad)
{
    for (unsigned i = 0; i < m_axes.size(); ++i)
        m_axes[i] = platformGamepad.axisValues()[i].value();
    for (unsigned i = 0; i < m_buttons.size(); ++i)
        m_buttons[i]->setValue(platformGamepad.buttonValues()[i].value());

    m_timestamp = platformGamepad.lastUpdateTime();
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
