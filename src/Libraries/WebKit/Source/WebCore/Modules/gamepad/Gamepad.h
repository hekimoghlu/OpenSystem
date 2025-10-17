/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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

#include "GamepadHapticEffectType.h"
#include <wtf/MonotonicTime.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Document;
class GamepadButton;
class GamepadHapticActuator;
class PlatformGamepad;

class Gamepad: public RefCountedAndCanMakeWeakPtr<Gamepad> {
public:
    static Ref<Gamepad> create(Document* document, const PlatformGamepad& platformGamepad)
    {
        return adoptRef(*new Gamepad(document, platformGamepad));
    }
    ~Gamepad();

    const String& id() const { return m_id; }
    unsigned index() const { return m_index; }
    const String& mapping() const { return m_mapping; }

    bool connected() const { return m_connected; }
    double timestamp() const { return m_timestamp.secondsSinceEpoch().seconds(); }
    const Vector<double>& axes() const;
    const Vector<Ref<GamepadButton>>& buttons() const;
    const GamepadHapticEffectTypeSet& supportedEffectTypes() const { return m_supportedEffectTypes; }

    void updateFromPlatformGamepad(const PlatformGamepad&);
    void setConnected(bool connected) { m_connected = connected; }

    GamepadHapticActuator* vibrationActuator() { return m_vibrationActuator.get(); }

private:
    Gamepad(Document*, const PlatformGamepad&);
    String m_id;
    unsigned m_index;
    bool m_connected;
    MonotonicTime m_timestamp;
    String m_mapping;
    GamepadHapticEffectTypeSet m_supportedEffectTypes;

    Vector<double> m_axes;
    Vector<Ref<GamepadButton>> m_buttons;

    RefPtr<GamepadHapticActuator> m_vibrationActuator;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
