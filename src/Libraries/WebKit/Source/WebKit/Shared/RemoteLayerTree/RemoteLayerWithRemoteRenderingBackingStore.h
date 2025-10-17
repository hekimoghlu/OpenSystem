/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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

#include "RemoteLayerBackingStore.h"
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class RemoteDisplayListRecorderProxy;
class RemoteImageBufferSetProxy;

class RemoteLayerWithRemoteRenderingBackingStore final : public RemoteLayerBackingStore {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerWithRemoteRenderingBackingStore);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteLayerWithRemoteRenderingBackingStore);
public:
    RemoteLayerWithRemoteRenderingBackingStore(PlatformCALayerRemote&);
    ~RemoteLayerWithRemoteRenderingBackingStore();

    bool isRemoteLayerWithRemoteRenderingBackingStore() const final { return true; }
    ProcessModel processModel() const final { return ProcessModel::Remote; }

    void prepareToDisplay() final;
    void clearBackingStore() final;
    void createContextAndPaintContents() final;

    RefPtr<RemoteImageBufferSetProxy> protectedBufferSet() { return m_bufferSet; }

    std::unique_ptr<ThreadSafeImageBufferSetFlusher> createFlusher(ThreadSafeImageBufferSetFlusher::FlushType) final;
    std::optional<ImageBufferBackendHandle> frontBufferHandle() const final { return std::exchange(const_cast<RemoteLayerWithRemoteRenderingBackingStore*>(this)->m_backendHandle, std::nullopt); }
#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    std::optional<ImageBufferBackendHandle> displayListHandle() const final;
#endif
    void encodeBufferAndBackendInfos(IPC::Encoder&) const final;
    std::optional<RemoteImageBufferSetIdentifier> bufferSetIdentifier() const final;

    void ensureBackingStore(const Parameters&) final;
    bool hasFrontBuffer() const final;
    bool frontBufferMayBeVolatile() const final;

    void setBufferCacheIdentifiers(BufferIdentifierSet&& identifiers)
    {
        m_bufferCacheIdentifiers = WTFMove(identifiers);
    }
    void setBackendHandle(std::optional<ImageBufferBackendHandle>&& backendHandle)
    {
        m_backendHandle = WTFMove(backendHandle);
    }

    void dump(WTF::TextStream&) const final;
private:
    RefPtr<RemoteImageBufferSetProxy> m_bufferSet;
    BufferIdentifierSet m_bufferCacheIdentifiers;
    std::optional<ImageBufferBackendHandle> m_backendHandle;
    bool m_cleared { true };
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::RemoteLayerWithRemoteRenderingBackingStore)
    static bool isType(const WebKit::RemoteLayerBackingStore& backingStore) { return backingStore.isRemoteLayerWithRemoteRenderingBackingStore(); }
SPECIALIZE_TYPE_TRAITS_END()
