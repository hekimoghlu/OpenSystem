/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include "Event.h"
#include "MediaEncryptedEventInit.h"

namespace JSC {
class ArrayBuffer;
}

namespace WebCore {

class MediaEncryptedEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaEncryptedEvent);
public:
    using Init = MediaEncryptedEventInit;

    static Ref<MediaEncryptedEvent> create(const AtomString& type, const MediaEncryptedEventInit& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new MediaEncryptedEvent(type, initializer, isTrusted));
    }

    virtual ~MediaEncryptedEvent();

    String initDataType() { return m_initDataType; }
    JSC::ArrayBuffer* initData() { return m_initData.get(); }

private:
    MediaEncryptedEvent(const AtomString&, const MediaEncryptedEventInit&, IsTrusted);

    String m_initDataType;
    RefPtr<JSC::ArrayBuffer> m_initData;
};

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
