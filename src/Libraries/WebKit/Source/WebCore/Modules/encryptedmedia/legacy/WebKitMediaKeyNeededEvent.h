/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "Event.h"
#include "WebKitMediaKeyError.h"

namespace WebCore {

class WebKitMediaKeyNeededEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebKitMediaKeyNeededEvent);
public:
    virtual ~WebKitMediaKeyNeededEvent();

    static Ref<WebKitMediaKeyNeededEvent> create(const AtomString& type, Uint8Array* initData)
    {
        return adoptRef(*new WebKitMediaKeyNeededEvent(type, initData));
    }

    struct Init : EventInit {
        RefPtr<Uint8Array> initData;
    };

    static Ref<WebKitMediaKeyNeededEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new WebKitMediaKeyNeededEvent(type, initializer, isTrusted));
    }

    Uint8Array* initData() const { return m_initData.get(); }

private:
    WebKitMediaKeyNeededEvent(const AtomString& type, Uint8Array* initData);
    WebKitMediaKeyNeededEvent(const AtomString& type, const Init&, IsTrusted);

    RefPtr<Uint8Array> m_initData;
};

} // namespace WebCore

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA)
