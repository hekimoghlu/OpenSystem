/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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

class BeforeTextInsertedEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(BeforeTextInsertedEvent);
public:
    virtual ~BeforeTextInsertedEvent();

    static Ref<BeforeTextInsertedEvent> create(const String& text)
    {
        return adoptRef(*new BeforeTextInsertedEvent(text));
    }

    const String& text() const { return m_text; }
    void setText(const String& s) { m_text = s; }

private:
    explicit BeforeTextInsertedEvent(const String&);
    bool isBeforeTextInsertedEvent() const override { return true; }

    String m_text;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(BeforeTextInsertedEvent)
