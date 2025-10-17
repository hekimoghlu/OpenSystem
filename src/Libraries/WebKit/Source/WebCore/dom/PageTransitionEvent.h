/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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

class PageTransitionEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PageTransitionEvent);
public:
    static Ref<PageTransitionEvent> create(const AtomString& type, bool persisted)
    {
        return adoptRef(*new PageTransitionEvent(type, persisted));
    }

    struct Init : EventInit {
        bool persisted { false };
    };

    static Ref<PageTransitionEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new PageTransitionEvent(type, initializer, isTrusted));
    }

    virtual ~PageTransitionEvent();

    bool persisted() const { return m_persisted; }

private:
    PageTransitionEvent(const AtomString& type, bool persisted);
    PageTransitionEvent(const AtomString&, const Init&, IsTrusted);

    bool m_persisted;
};

} // namespace WebCore
