/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#include "SharedGamepadValue.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/MonotonicTime.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/WeakHashMap.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class PlatformGamepad;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::PlatformGamepad> : std::true_type { };
}

namespace WebCore {

struct GamepadEffectParameters;

class PlatformGamepad : public CanMakeWeakPtr<PlatformGamepad> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PlatformGamepad);
public:
    virtual ~PlatformGamepad() = default;

    const String& id() const { return m_id; }
    const String& mapping() const { return m_mapping; }
    unsigned index() const { return m_index; }
    virtual MonotonicTime lastUpdateTime() const { return m_lastUpdateTime; }
    MonotonicTime connectTime() const { return m_connectTime; }
    const GamepadHapticEffectTypeSet& supportedEffectTypes() const { return m_supportedEffectTypes; }
    
    virtual const Vector<SharedGamepadValue>& axisValues() const = 0;
    virtual const Vector<SharedGamepadValue>& buttonValues() const = 0;
    virtual void playEffect(GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&& completionHandler) { completionHandler(false); }
    virtual void stopEffects(CompletionHandler<void()>&& completionHandler) { completionHandler(); }

    virtual ASCIILiteral source() const { return "Unknown"_s; }

protected:
    explicit PlatformGamepad(unsigned index)
        : m_index(index)
    {
    }

    String m_id;
    String m_mapping;
    unsigned m_index;
    MonotonicTime m_lastUpdateTime;
    MonotonicTime m_connectTime;
    GamepadHapticEffectTypeSet m_supportedEffectTypes;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
