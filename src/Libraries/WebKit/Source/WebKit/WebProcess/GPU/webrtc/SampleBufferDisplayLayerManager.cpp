/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
#include "config.h"
#include "SampleBufferDisplayLayerManager.h"

#include "Decoder.h"
#include "GPUProcessConnection.h"
#include <WebCore/IntSize.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM)

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(SampleBufferDisplayLayerManager);

SampleBufferDisplayLayerManager::SampleBufferDisplayLayerManager(GPUProcessConnection& gpuProcessConnection)
    : m_gpuProcessConnection(gpuProcessConnection)
{
}

void SampleBufferDisplayLayerManager::didReceiveLayerMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    if (ObjectIdentifier<SampleBufferDisplayLayerIdentifierType>::isValidIdentifier(decoder.destinationID())) {
        if (auto* layer = m_layers.get(ObjectIdentifier<SampleBufferDisplayLayerIdentifierType>(decoder.destinationID())).get())
            layer->didReceiveMessage(connection, decoder);
    }
}

RefPtr<WebCore::SampleBufferDisplayLayer> SampleBufferDisplayLayerManager::createLayer(WebCore::SampleBufferDisplayLayerClient& client)
{
    auto layer = SampleBufferDisplayLayer::create(*this, client);
    m_layers.add(layer->identifier(), layer.get());
    return layer;
}

void SampleBufferDisplayLayerManager::addLayer(SampleBufferDisplayLayer& layer)
{
    ASSERT(!m_layers.contains(layer.identifier()));
    m_layers.add(layer.identifier(), layer);
}

void SampleBufferDisplayLayerManager::removeLayer(SampleBufferDisplayLayer& layer)
{
    ASSERT(m_layers.contains(layer.identifier()));
    m_layers.remove(layer.identifier());
}

void SampleBufferDisplayLayerManager::ref() const
{
    m_gpuProcessConnection.get()->ref();
}

void SampleBufferDisplayLayerManager::deref() const
{
    m_gpuProcessConnection.get()->deref();
}

}

#endif
