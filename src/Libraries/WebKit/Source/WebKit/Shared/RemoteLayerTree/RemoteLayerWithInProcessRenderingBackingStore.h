/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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

#include "RemoteImageBufferSet.h"
#include "RemoteLayerBackingStore.h"
#include <WebCore/DynamicContentScalingResourceCache.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class RemoteLayerWithInProcessRenderingBackingStore final : public RemoteLayerBackingStore {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerWithInProcessRenderingBackingStore);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteLayerWithInProcessRenderingBackingStore);
public:
    using RemoteLayerBackingStore::RemoteLayerBackingStore;

    bool isRemoteLayerWithInProcessRenderingBackingStore() const final { return true; }
    ProcessModel processModel() const final { return ProcessModel::InProcess; }

    void prepareToDisplay() final;
    void createContextAndPaintContents() final;
    std::unique_ptr<ThreadSafeImageBufferSetFlusher> createFlusher(ThreadSafeImageBufferSetFlusher::FlushType) final;

    void clearBackingStore() final;

    bool setBufferVolatile(BufferType, bool forcePurge = false);

    std::optional<ImageBufferBackendHandle> frontBufferHandle() const final;
#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    std::optional<ImageBufferBackendHandle> displayListHandle() const final;
#endif
    void encodeBufferAndBackendInfos(IPC::Encoder&) const final;

    void dump(WTF::TextStream&) const final;

private:
    RefPtr<WebCore::ImageBuffer> allocateBuffer();
    void ensureFrontBuffer();
    bool hasFrontBuffer() const final;
    bool frontBufferMayBeVolatile() const final;

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    WebCore::DynamicContentScalingResourceCache ensureDynamicContentScalingResourceCache();
#endif

    struct Buffer {
        RefPtr<WebCore::ImageBuffer> imageBuffer;
        bool isCleared { false };

        explicit operator bool() const
        {
            return !!imageBuffer;
        }

        void discard();
    };

    // Returns true if it was able to fulfill the request. This can fail when trying to mark an in-use surface as volatile.
    bool setBufferVolatile(RefPtr<WebCore::ImageBuffer>&, bool forcePurge = false);
    WebCore::SetNonVolatileResult setBufferNonVolatile(Buffer&);

    ImageBufferSet m_bufferSet;

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    WebCore::DynamicContentScalingResourceCache m_dynamicContentScalingResourceCache;
#endif
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::RemoteLayerWithInProcessRenderingBackingStore)
    static bool isType(const WebKit::RemoteLayerBackingStore& backingStore) { return backingStore.isRemoteLayerWithInProcessRenderingBackingStore(); }
SPECIALIZE_TYPE_TRAITS_END()
