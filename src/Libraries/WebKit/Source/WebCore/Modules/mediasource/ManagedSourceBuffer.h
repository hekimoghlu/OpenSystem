/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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

#if ENABLE(MEDIA_SOURCE)

#include "SourceBuffer.h"

namespace WebCore {

class ManagedMediaSource;

class ManagedSourceBuffer final : public SourceBuffer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ManagedSourceBuffer);
public:
    static Ref<ManagedSourceBuffer> create(Ref<SourceBufferPrivate>&&, ManagedMediaSource&);
    ~ManagedSourceBuffer();

    bool isManaged() const final { return true; }

private:
    ManagedSourceBuffer(Ref<SourceBufferPrivate>&&, ManagedMediaSource&);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ManagedSourceBuffer)
    static bool isType(const WebCore::SourceBuffer& buffer) { return buffer.isManaged(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(MEDIA_SOURCE)
