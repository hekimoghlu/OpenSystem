/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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

#if ENABLE(MEDIA_RECORDER)

#include "Event.h"

namespace WebCore {
    
class Blob;
    
class BlobEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(BlobEvent);
public:
    struct Init : EventInit {
        RefPtr<Blob> data;
        double timecode;
    };
    
    static Ref<BlobEvent> create(const AtomString&, Init&&, IsTrusted = IsTrusted::No);

    Blob& data() const { return m_blob.get(); }
    double timecode() const { return m_timecode; }

private:
    BlobEvent(const AtomString&, Init&&, IsTrusted);
    BlobEvent(const AtomString&, CanBubble, IsCancelable, Ref<Blob>&&);

    Ref<Blob> m_blob;
    double m_timecode { 0 };
};
    
} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
