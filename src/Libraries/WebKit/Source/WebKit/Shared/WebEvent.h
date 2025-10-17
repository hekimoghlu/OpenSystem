/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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

// FIXME: We should probably move to making the WebCore/PlatformFooEvents trivial classes so that
// we can use them as the event type.

#include "WebEvent.h"
#include "WebEventModifier.h"
#include "WebEventType.h"
#include <wtf/CheckedPtr.h>
#include <wtf/OptionSet.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UUID.h>
#include <wtf/WallTime.h>
#include <wtf/text/WTFString.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

class WebEvent : public CanMakeCheckedPtr<WebEvent> {
    WTF_MAKE_TZONE_ALLOCATED(WebEvent);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebEvent);
public:
    WebEvent(WebEventType, OptionSet<WebEventModifier>, WallTime timestamp, WTF::UUID authorizationToken);
    WebEvent(WebEventType, OptionSet<WebEventModifier>, WallTime timestamp);

    WebEventType type() const { return m_type; }

    bool shiftKey() const { return m_modifiers.contains(WebEventModifier::ShiftKey); }
    bool controlKey() const { return m_modifiers.contains(WebEventModifier::ControlKey); }
    bool altKey() const { return m_modifiers.contains(WebEventModifier::AltKey); }
    bool metaKey() const { return m_modifiers.contains(WebEventModifier::MetaKey); }
    bool capsLockKey() const { return m_modifiers.contains(WebEventModifier::CapsLockKey); }

    OptionSet<WebEventModifier> modifiers() const { return m_modifiers; }

    WallTime timestamp() const { return m_timestamp; }

    bool isActivationTriggeringEvent() const;
    WTF::UUID authorizationToken() const { return m_authorizationToken; }

private:
    WebEventType m_type;
    OptionSet<WebEventModifier> m_modifiers;
    WallTime m_timestamp;
    WTF::UUID m_authorizationToken;
};

WTF::TextStream& operator<<(WTF::TextStream&, WebEventType);

} // namespace WebKit
