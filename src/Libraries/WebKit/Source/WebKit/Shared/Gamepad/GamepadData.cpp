/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#include "GamepadData.h"

#if ENABLE(GAMEPAD)

#include "ArgumentCoders.h"
#include <wtf/text/StringBuilder.h>

using WebCore::SharedGamepadValue;

namespace WebKit {

GamepadData::GamepadData(unsigned index, const String& id, const String& mapping, const Vector<SharedGamepadValue>& axisValues, const Vector<SharedGamepadValue>& buttonValues, MonotonicTime lastUpdateTime, const WebCore::GamepadHapticEffectTypeSet& supportedEffectTypes)
    : m_index(index)
    , m_id(id)
    , m_mapping(mapping)
    , m_axisValues(WTF::map(axisValues, [](const auto& value) { return value.value(); }))
    , m_buttonValues(WTF::map(buttonValues, [](const auto& value) { return value.value(); }))
    , m_lastUpdateTime(lastUpdateTime)
    , m_supportedEffectTypes(supportedEffectTypes)
{
}

GamepadData::GamepadData(unsigned index, String&& id, String&& mapping, Vector<double>&& axisValues, Vector<double>&& buttonValues, MonotonicTime lastUpdateTime, WebCore::GamepadHapticEffectTypeSet&& supportedEffectTypes)
    : m_index(index)
    , m_id(WTFMove(id))
    , m_mapping(WTFMove(mapping))
    , m_axisValues(WTFMove(axisValues))
    , m_buttonValues(WTFMove(buttonValues))
    , m_lastUpdateTime(lastUpdateTime)
    , m_supportedEffectTypes(WTFMove(supportedEffectTypes))
{
}

#if !LOG_DISABLED

String GamepadData::loggingString() const
{
    StringBuilder builder;

    builder.append(m_axisValues.size(), " axes, "_s, m_buttonValues.size(), " buttons\n"_s);

    for (size_t i = 0; i < m_axisValues.size(); ++i)
        builder.append(" Axis "_s, i, ": "_s, m_axisValues[i]);

    builder.append('\n');

    for (size_t i = 0; i < m_buttonValues.size(); ++i)
        builder.append(" Button "_s, i, ": "_s, FormattedNumber::fixedPrecision(m_buttonValues[i]));

    return builder.toString();
}

#endif

} // namespace WebKit

#endif // ENABLE(GAMEPAD)
