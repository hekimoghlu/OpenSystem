/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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

#include "Event.h"

namespace WebCore {

class OverflowEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(OverflowEvent);
public:
    enum orientType {
        HORIZONTAL = 0,
        VERTICAL   = 1,
        BOTH       = 2
    };

    static Ref<OverflowEvent> create(bool horizontalOverflowChanged, bool horizontalOverflow, bool verticalOverflowChanged, bool verticalOverflow)
    {
        return adoptRef(*new OverflowEvent(horizontalOverflowChanged, horizontalOverflow, verticalOverflowChanged, verticalOverflow));
    }

    static Ref<OverflowEvent> createForBindings()
    {
        return adoptRef(*new OverflowEvent);
    }

    struct Init : EventInit {
        unsigned short orient { 0 };
        bool horizontalOverflow { false };
        bool verticalOverflow { false };
    };

    static Ref<OverflowEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new OverflowEvent(type, initializer, isTrusted));
    }

    WEBCORE_EXPORT void initOverflowEvent(unsigned short orient, bool horizontalOverflow, bool verticalOverflow);

    unsigned short orient() const { return m_orient; }
    bool horizontalOverflow() const { return m_horizontalOverflow; }
    bool verticalOverflow() const { return m_verticalOverflow; }

private:
    OverflowEvent();
    OverflowEvent(bool horizontalOverflowChanged, bool horizontalOverflow, bool verticalOverflowChanged, bool verticalOverflow);
    OverflowEvent(const AtomString&, const Init&, IsTrusted);

    unsigned short m_orient;
    bool m_horizontalOverflow;
    bool m_verticalOverflow;
};

} // namespace WebCore
