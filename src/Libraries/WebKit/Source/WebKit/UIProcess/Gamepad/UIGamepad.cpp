/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
#include "UIGamepad.h"

#if ENABLE(GAMEPAD)

#include "GamepadData.h"
#include <WebCore/PlatformGamepad.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(UIGamepad);

UIGamepad::UIGamepad(WebCore::PlatformGamepad& platformGamepad)
    : m_index(platformGamepad.index())
    , m_id(platformGamepad.id())
    , m_mapping(platformGamepad.mapping())
    , m_axisValues(platformGamepad.axisValues().size())
    , m_buttonValues(platformGamepad.buttonValues().size())
    , m_lastUpdateTime(platformGamepad.lastUpdateTime())
    , m_supportedEffectTypes(platformGamepad.supportedEffectTypes())
{
    updateFromPlatformGamepad(platformGamepad);
}

void UIGamepad::updateFromPlatformGamepad(WebCore::PlatformGamepad& platformGamepad)
{
    ASSERT(m_index == platformGamepad.index());
    ASSERT(m_axisValues.size() == platformGamepad.axisValues().size());
    ASSERT(m_buttonValues.size() == platformGamepad.buttonValues().size());

    m_axisValues = platformGamepad.axisValues();
    m_buttonValues = platformGamepad.buttonValues();
    m_lastUpdateTime = platformGamepad.lastUpdateTime();
}

GamepadData UIGamepad::gamepadData() const
{
    return { m_index, m_id, m_mapping, m_axisValues, m_buttonValues, m_lastUpdateTime, m_supportedEffectTypes };
}

}

#endif // ENABLE(GAMEPAD)
