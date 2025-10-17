/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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
#import "config.h"
#import "RemoteLayerWithRemoteRenderingBackingStore.h"

#import "PlatformCALayerRemote.h"
#import "RemoteImageBufferSetProxy.h"
#import "RemoteLayerBackingStoreCollection.h"
#import "RemoteLayerTreeContext.h"
#import "RemoteRenderingBackendProxy.h"
#import "SwapBuffersDisplayRequirement.h"
#import <WebCore/PlatformCALayerClient.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteLayerWithRemoteRenderingBackingStore);

using namespace WebCore;

RemoteLayerWithRemoteRenderingBackingStore::RemoteLayerWithRemoteRenderingBackingStore(PlatformCALayerRemote& layer)
    : RemoteLayerBackingStore(layer)
{
    RefPtr collection = backingStoreCollection();
    if (!collection) {
        ASSERT_NOT_REACHED();
        return;
    }

    m_bufferSet = collection->layerTreeContext().ensureRemoteRenderingBackendProxy().createRemoteImageBufferSet();
}

RemoteLayerWithRemoteRenderingBackingStore::~RemoteLayerWithRemoteRenderingBackingStore()
{
    if (m_bufferSet)
        m_bufferSet->close();
}

bool RemoteLayerWithRemoteRenderingBackingStore::hasFrontBuffer() const
{
    return m_contentsBufferHandle || !m_cleared;
}

bool RemoteLayerWithRemoteRenderingBackingStore::frontBufferMayBeVolatile() const
{
    if (!m_bufferSet)
        return false;
    return m_bufferSet->requestedVolatility().contains(BufferInSetType::Front);
}

void RemoteLayerWithRemoteRenderingBackingStore::prepareToDisplay()
{
    m_contentsBufferHandle = std::nullopt;
}

void RemoteLayerWithRemoteRenderingBackingStore::clearBackingStore()
{
    m_contentsBufferHandle = std::nullopt;
    m_cleared = true;
}

std::unique_ptr<ThreadSafeImageBufferSetFlusher> RemoteLayerWithRemoteRenderingBackingStore::createFlusher(ThreadSafeImageBufferSetFlusher::FlushType flushType)
{
    if (!m_bufferSet)
        return { };
    return m_bufferSet->flushFrontBufferAsync(flushType);
}

void RemoteLayerWithRemoteRenderingBackingStore::createContextAndPaintContents()
{
    auto bufferSet = protectedBufferSet();
    if (!bufferSet)
        return;

    if (!bufferSet->hasContext()) {
        // The platform layer delegates display or bufferSet does not have a working connection to GPUP anymore.
        return;
    }

    drawInContext(bufferSet->context());
    m_cleared = false;
}

void RemoteLayerWithRemoteRenderingBackingStore::ensureBackingStore(const Parameters& parameters)
{
    if (m_parameters == parameters)
        return;

    m_parameters = parameters;
    m_cleared = true;
    if (m_bufferSet) {
        auto renderingMode = type() == RemoteLayerBackingStore::Type::IOSurface ? RenderingMode::Accelerated : RenderingMode::Unaccelerated;
        auto renderingPurpose = m_layer->containsBitmapOnly() ? WebCore::RenderingPurpose::BitmapOnlyLayerBacking : WebCore::RenderingPurpose::LayerBacking;
        m_bufferSet->setConfiguration(size(), scale(), colorSpace(), contentsFormat(), pixelFormat(), renderingMode, renderingPurpose);
    }
}

void RemoteLayerWithRemoteRenderingBackingStore::encodeBufferAndBackendInfos(IPC::Encoder& encoder) const
{
    auto encodeBuffer = [&](const  std::optional<WebCore::RenderingResourceIdentifier>& bufferIdentifier) {
        if (bufferIdentifier) {
            encoder << std::optional { BufferAndBackendInfo { *bufferIdentifier, m_bufferSet->generation() } };
            return;
        }

        encoder << std::optional<BufferAndBackendInfo>();
    };

    encodeBuffer(m_bufferCacheIdentifiers.front);
    encodeBuffer(m_bufferCacheIdentifiers.back);
    encodeBuffer(m_bufferCacheIdentifiers.secondaryBack);
}

std::optional<RemoteImageBufferSetIdentifier> RemoteLayerWithRemoteRenderingBackingStore::bufferSetIdentifier() const
{
    if (!m_bufferSet)
        return std::nullopt;
    return m_bufferSet->identifier();
}

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
std::optional<ImageBufferBackendHandle> RemoteLayerWithRemoteRenderingBackingStore::displayListHandle() const
{
    return m_bufferSet ? m_bufferSet->dynamicContentScalingDisplayList() : std::nullopt;
}
#endif

void RemoteLayerWithRemoteRenderingBackingStore::dump(WTF::TextStream& ts) const
{
    ts.dumpProperty("buffer set", m_bufferSet);
    ts.dumpProperty("cache identifiers", m_bufferCacheIdentifiers);
    ts.dumpProperty("is opaque", isOpaque());
}

} // namespace WebKit

