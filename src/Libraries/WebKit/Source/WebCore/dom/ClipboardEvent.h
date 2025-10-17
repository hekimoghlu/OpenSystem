/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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

class DataTransfer;

class ClipboardEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ClipboardEvent);
public:
    virtual ~ClipboardEvent();

    struct Init : EventInit {
        RefPtr<DataTransfer> clipboardData;
    };

    static Ref<ClipboardEvent> create(const AtomString& type, Ref<DataTransfer>&& dataTransfer)
    {
        return adoptRef(*new ClipboardEvent(type, WTFMove(dataTransfer)));
    }

    static Ref<ClipboardEvent> create(const AtomString& type, const Init& init)
    {
        return adoptRef(*new ClipboardEvent(type, init));
    }

    DataTransfer* clipboardData() const { return m_clipboardData.get(); }

private:
    ClipboardEvent(const AtomString& type, Ref<DataTransfer>&&);
    ClipboardEvent(const AtomString& type, const Init&);

    bool isClipboardEvent() const final;

    RefPtr<DataTransfer> m_clipboardData;
};

} // namespace WebCore
