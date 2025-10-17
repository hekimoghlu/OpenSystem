/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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

class ProgressEvent : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ProgressEvent);
public:
    static Ref<ProgressEvent> create(const AtomString& type, bool lengthComputable, unsigned long long loaded, unsigned long long total)
    {
        return adoptRef(*new ProgressEvent(EventInterfaceType::ProgressEvent, type, lengthComputable, loaded, total));
    }

    struct Init : EventInit {
        bool lengthComputable { false };
        unsigned long long loaded { 0 };
        unsigned long long total { 0 };
    };

    static Ref<ProgressEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new ProgressEvent(EventInterfaceType::ProgressEvent, type, initializer, isTrusted));
    }

    bool lengthComputable() const { return m_lengthComputable; }
    unsigned long long loaded() const { return m_loaded; }
    unsigned long long total() const { return m_total; }

protected:
    ProgressEvent(enum EventInterfaceType, const AtomString& type, bool lengthComputable, unsigned long long loaded, unsigned long long total);
    ProgressEvent(enum EventInterfaceType, const AtomString&, const Init&, IsTrusted);

private:
    bool m_lengthComputable;
    unsigned long long m_loaded;
    unsigned long long m_total;
};

} // namespace WebCore
