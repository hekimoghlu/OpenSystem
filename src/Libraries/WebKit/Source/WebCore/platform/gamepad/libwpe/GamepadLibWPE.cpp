/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
#include "GamepadLibWPE.h"

#if ENABLE(GAMEPAD) && USE(LIBWPE)

#include "GamepadProviderLibWPE.h"
#include <wpe/wpe.h>

namespace WebCore {

GamepadLibWPE::GamepadLibWPE(struct wpe_gamepad_provider* provider, uintptr_t gamepadId, unsigned index)
    : PlatformGamepad(index)
    , m_buttonValues(WPE_GAMEPAD_BUTTON_COUNT)
    , m_axisValues(WPE_GAMEPAD_AXIS_COUNT)
    , m_gamepad(wpe_gamepad_create(provider, gamepadId), wpe_gamepad_destroy)
{
    ASSERT(m_gamepad);

    m_connectTime = m_lastUpdateTime = MonotonicTime::now();

    m_id = String::fromUTF8(wpe_gamepad_get_id(m_gamepad.get()));
    m_mapping = String::fromUTF8("standard");

    static const struct wpe_gamepad_client_interface s_client = {
        // button_event
        [](void* data, enum wpe_gamepad_button button, bool pressed) {
            auto& self = *static_cast<GamepadLibWPE*>(data);
            self.buttonPressedOrReleased(static_cast<unsigned>(button), pressed);
        },
        // axis_event
        [](void* data, enum wpe_gamepad_axis axis, double value) {
            auto& self = *static_cast<GamepadLibWPE*>(data);
            self.absoluteAxisChanged(static_cast<unsigned>(axis), value);
        },
        nullptr, nullptr, nullptr,
    };
    wpe_gamepad_set_client(m_gamepad.get(), &s_client, this);
}

GamepadLibWPE::~GamepadLibWPE()
{
    wpe_gamepad_set_client(m_gamepad.get(), nullptr, nullptr);
}

void GamepadLibWPE::buttonPressedOrReleased(unsigned button, bool pressed)
{
    m_lastUpdateTime = MonotonicTime::now();
    m_buttonValues[button].setValue(pressed ? 1.0 : 0.0);

    GamepadProviderLibWPE::singleton().scheduleInputNotification(*this, pressed ? GamepadProviderLibWPE::ShouldMakeGamepadsVisible::Yes : GamepadProviderLibWPE::ShouldMakeGamepadsVisible::No);
}

void GamepadLibWPE::absoluteAxisChanged(unsigned axis, double value)
{
    m_lastUpdateTime = MonotonicTime::now();
    m_axisValues[axis].setValue(value);

    GamepadProviderLibWPE::singleton().scheduleInputNotification(*this, GamepadProviderLibWPE::ShouldMakeGamepadsVisible::Yes);
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && USE(LIBWPE)
