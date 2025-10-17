/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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

#if ENABLE(VIDEO)

#include "ContentType.h"
#include "MediaPromiseTypes.h"
#include "PlatformMediaResourceLoader.h"
#include <wtf/NativePromise.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class MediaResourceSniffer final : public PlatformMediaResourceClient {
    WTF_MAKE_TZONE_ALLOCATED(MediaResourceSniffer);
public:
    static Ref<MediaResourceSniffer> create(PlatformMediaResourceLoader&, ResourceRequest&&, std::optional<size_t> maxSize);
    ~MediaResourceSniffer();

    using Promise = NativePromise<ContentType, PlatformMediaError>;
    Promise& promise() const;
    void cancel();

private:
    MediaResourceSniffer(Ref<PlatformMediaResource>&&, size_t);
    MediaResourceSniffer();

    void dataReceived(PlatformMediaResource&, const SharedBuffer&) final;
    void loadFailed(PlatformMediaResource&, const ResourceError&) final;
    void loadFinished(PlatformMediaResource&, const NetworkLoadMetrics&) final;

    RefPtr<PlatformMediaResource> m_resource;
    const size_t m_maxSize;
    size_t m_received { 0 };
    Promise::Producer m_producer;
    SharedBufferBuilder m_content;
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
