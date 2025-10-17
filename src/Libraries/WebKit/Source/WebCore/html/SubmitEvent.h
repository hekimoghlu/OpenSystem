/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
#include "EventInit.h"
#include <wtf/Forward.h>

namespace WebCore {

class HTMLElement;

class SubmitEvent : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SubmitEvent);
public:
    struct Init : EventInit {
        RefPtr<HTMLElement> submitter;
    };

    static Ref<SubmitEvent> create(const AtomString& type, Init&&);
    static Ref<SubmitEvent> create(RefPtr<HTMLElement>&& submitter);

    HTMLElement* submitter() const { return m_submitter.get(); }

private:
    SubmitEvent(const AtomString& type, Init&&);
    explicit SubmitEvent(RefPtr<HTMLElement>&& submitter);

    RefPtr<HTMLElement> m_submitter;
};

} // namespace WebCore
