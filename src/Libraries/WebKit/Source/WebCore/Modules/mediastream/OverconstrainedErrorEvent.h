/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 20, 2023.
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

#if ENABLE(MEDIA_STREAM)

#include "Event.h"
#include "OverconstrainedError.h"
#include <wtf/text/AtomString.h>

namespace WebCore {

class OverconstrainedErrorEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(OverconstrainedErrorEvent);
public:
    virtual ~OverconstrainedErrorEvent() = default;

    static Ref<OverconstrainedErrorEvent> create(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, OverconstrainedError* error)
    {
        return adoptRef(*new OverconstrainedErrorEvent(type, canBubble, cancelable, error));
    }

    struct Init : EventInit {
        RefPtr<OverconstrainedError> error;
    };

    static Ref<OverconstrainedErrorEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new OverconstrainedErrorEvent(type, initializer, isTrusted));
    }

    OverconstrainedError* error() const { return m_error.get(); }

private:
    explicit OverconstrainedErrorEvent(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, OverconstrainedError* error)
        : Event(EventInterfaceType::OverconstrainedErrorEvent, type, canBubble, cancelable)
        , m_error(error)
    {
    }
    OverconstrainedErrorEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
        : Event(EventInterfaceType::OverconstrainedErrorEvent, type, initializer, isTrusted)
        , m_error(initializer.error)
    {
    }

    RefPtr<OverconstrainedError> m_error;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
