/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#include "RemoteImageDecoderAVFManager.h"

#if ENABLE(GPU_PROCESS) && HAVE(AVASSETREADER)

#include "GPUProcessConnection.h"
#include "RemoteImageDecoderAVF.h"
#include "RemoteImageDecoderAVFManagerMessages.h"
#include "RemoteImageDecoderAVFProxyMessages.h"
#include "SharedBufferReference.h"
#include "WebProcess.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteImageDecoderAVFManager);

RefPtr<RemoteImageDecoderAVF> RemoteImageDecoderAVFManager::createImageDecoder(FragmentedSharedBuffer& data, const String& mimeType, AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
{
    if (!WebProcess::singleton().mediaPlaybackEnabled())
        return nullptr;

    auto sendResult = ensureGPUProcessConnection().connection().sendSync(Messages::RemoteImageDecoderAVFProxy::CreateDecoder(IPC::SharedBufferReference(data), mimeType), 0);
    auto [imageDecoderIdentifier] = sendResult.takeReplyOr(std::nullopt);
    if (!imageDecoderIdentifier)
        return nullptr;

    auto remoteImageDecoder = RemoteImageDecoderAVF::create(*this, *imageDecoderIdentifier, data, mimeType);
    m_remoteImageDecoders.add(*imageDecoderIdentifier, remoteImageDecoder);

    return remoteImageDecoder;
}

void RemoteImageDecoderAVFManager::deleteRemoteImageDecoder(const ImageDecoderIdentifier& identifier)
{
    m_remoteImageDecoders.take(identifier);
    if (auto gpuProcessConnection = m_gpuProcessConnection.get())
        gpuProcessConnection->connection().send(Messages::RemoteImageDecoderAVFProxy::DeleteDecoder(identifier), 0);
}

Ref<RemoteImageDecoderAVFManager> RemoteImageDecoderAVFManager::create()
{
    return adoptRef(*new RemoteImageDecoderAVFManager);
}

RemoteImageDecoderAVFManager::RemoteImageDecoderAVFManager() = default;

RemoteImageDecoderAVFManager::~RemoteImageDecoderAVFManager()
{
    if (auto gpuProcessConnection = m_gpuProcessConnection.get())
        gpuProcessConnection->messageReceiverMap().removeMessageReceiver(Messages::RemoteImageDecoderAVFManager::messageReceiverName());
}

void RemoteImageDecoderAVFManager::gpuProcessConnectionDidClose(GPUProcessConnection& connection)
{
    ASSERT(m_gpuProcessConnection.get() == &connection);
    if (auto gpuProcessConnection = m_gpuProcessConnection.get())
        gpuProcessConnection->messageReceiverMap().removeMessageReceiver(Messages::RemoteImageDecoderAVFManager::messageReceiverName());
    m_gpuProcessConnection = nullptr;
    // FIXME: Do we need to do more when m_remoteImageDecoders is not empty to re-create them?
}

GPUProcessConnection& RemoteImageDecoderAVFManager::ensureGPUProcessConnection()
{
    auto gpuProcessConnection = m_gpuProcessConnection.get();
    if (!gpuProcessConnection) {
        gpuProcessConnection = &WebProcess::singleton().ensureGPUProcessConnection();
        m_gpuProcessConnection = gpuProcessConnection;
        gpuProcessConnection->addClient(*this);
        gpuProcessConnection->messageReceiverMap().addMessageReceiver(Messages::RemoteImageDecoderAVFManager::messageReceiverName(), *this);
    }
    return *gpuProcessConnection;
}

void RemoteImageDecoderAVFManager::setUseGPUProcess(bool useGPUProcess)
{
    if (!useGPUProcess) {
        ImageDecoder::resetFactories();
        return;
    }

    ImageDecoder::clearFactories();
    ImageDecoder::installFactory({
        RemoteImageDecoderAVF::supportsMediaType,
        RemoteImageDecoderAVF::canDecodeType,
        [this](FragmentedSharedBuffer& data, const String& mimeType, AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption) {
            return createImageDecoder(data, mimeType, alphaOption, gammaAndColorProfileOption);
        }
    });
}

void RemoteImageDecoderAVFManager::encodedDataStatusChanged(const ImageDecoderIdentifier& identifier, size_t frameCount, const WebCore::IntSize& size, bool hasTrack)
{
    if (!m_remoteImageDecoders.contains(identifier))
        return;

    auto remoteImageDecoder = m_remoteImageDecoders.get(identifier);
    if (!remoteImageDecoder)
        return;

    remoteImageDecoder->encodedDataStatusChanged(frameCount, size, hasTrack);
}

}

#endif
