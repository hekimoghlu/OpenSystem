/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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

#if ENABLE(GAMEPAD)

#include <WebCore/GamepadHapticEffectType.h>
#include <WebCore/SharedGamepadValue.h>
#include <wtf/MonotonicTime.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class GamepadData {
public:
    GamepadData(unsigned index, const String& id, const String& mapping, const Vector<WebCore::SharedGamepadValue>& axisValues, const Vector<WebCore::SharedGamepadValue>& buttonValues, MonotonicTime lastUpdateTime, const WebCore::GamepadHapticEffectTypeSet& supportedEffectTypes);
    
    GamepadData(unsigned index, String&& id, String&& mapping, Vector<double>&& axisValues, Vector<double>&& buttonValues, MonotonicTime lastUpdateTime, WebCore::GamepadHapticEffectTypeSet&& supportedEffectTypes);

    MonotonicTime lastUpdateTime() const { return m_lastUpdateTime; }
    unsigned index() const { return m_index; }
    const String& id() const { return m_id; }
    const String& mapping() const { return m_mapping; }
    const Vector<double>& axisValues() const { return m_axisValues; }
    const Vector<double>& buttonValues() const { return m_buttonValues; }
    const WebCore::GamepadHapticEffectTypeSet& supportedEffectTypes() const { return m_supportedEffectTypes; }

private:
    unsigned m_index;
    String m_id;
    String m_mapping;
    Vector<double> m_axisValues;
    Vector<double> m_buttonValues;
    MonotonicTime m_lastUpdateTime;
    WebCore::GamepadHapticEffectTypeSet m_supportedEffectTypes;

#if !LOG_DISABLED
    String loggingString() const;
#endif
};

} // namespace WebKit

#endif // ENABLE(GAMEPAD)
