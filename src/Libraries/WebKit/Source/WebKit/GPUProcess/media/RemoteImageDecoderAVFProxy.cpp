/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#include "RemoteImageDecoderAVFProxy.h"

#if ENABLE(GPU_PROCESS) && HAVE(AVASSETREADER)

#include "GPUConnectionToWebProcess.h"
#include "GPUProcess.h"
#include "RemoteImageDecoderAVFManagerMessages.h"
#include "RemoteImageDecoderAVFProxyMessages.h"
#include "SharedBufferReference.h"
#include <CoreGraphics/CGImage.h>
#include <WebCore/GraphicsContext.h>
#include <WebCore/ImageDecoderAVFObjC.h>
#include <wtf/Scope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

RemoteImageDecoderAVFProxy::RemoteImageDecoderAVFProxy(GPUConnectionToWebProcess& connectionToWebProcess)
    : m_connectionToWebProcess(connectionToWebProcess)
    , m_resourceOwner(connectionToWebProcess.webProcessIdentity())
{
}

void RemoteImageDecoderAVFProxy::ref() const
{
    m_connectionToWebProcess.get()->ref();
}

void RemoteImageDecoderAVFProxy::deref() const
{
    m_connectionToWebProcess.get()->deref();
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteImageDecoderAVFProxy);

void RemoteImageDecoderAVFProxy::createDecoder(const IPC::SharedBufferReference& data, const String& mimeType, CompletionHandler<void(std::optional<ImageDecoderIdentifier>&&)>&& completionHandler)
{
    auto imageDecoder = ImageDecoderAVFObjC::create(data.isNull() ? SharedBuffer::create() : data.unsafeBuffer().releaseNonNull(), mimeType, AlphaOption::Premultiplied, GammaAndColorProfileOption::Ignored, m_resourceOwner);

    std::optional<ImageDecoderIdentifier> imageDecoderIdentifier;
    if (!imageDecoder)
        return completionHandler(WTFMove(imageDecoderIdentifier));

    auto identifier = ImageDecoderIdentifier::generate();
    m_imageDecoders.add(identifier, imageDecoder.copyRef());

    imageDecoder->setEncodedDataStatusChangeCallback([proxy = WeakPtr { *this },  identifier](auto) mutable {
        if (RefPtr protectedProxy = proxy.get())
            protectedProxy->encodedDataStatusChanged(identifier);
    });

    imageDecoderIdentifier = identifier;
    completionHandler(WTFMove(imageDecoderIdentifier));
}

void RemoteImageDecoderAVFProxy::deleteDecoder(ImageDecoderIdentifier identifier)
{
    ASSERT(m_imageDecoders.contains(identifier));
    if (!m_imageDecoders.contains(identifier))
        return;

    m_imageDecoders.take(identifier);
    RefPtr connection = m_connectionToWebProcess.get();
    if (!connection)
        return;
    if (allowsExitUnderMemoryPressure())
        connection->protectedGPUProcess()->tryExitIfUnusedAndUnderMemoryPressure();
}

void RemoteImageDecoderAVFProxy::encodedDataStatusChanged(ImageDecoderIdentifier identifier)
{
    RefPtr connection = m_connectionToWebProcess.get();
    if (!connection)
        return;

    RefPtr imageDecoder = m_imageDecoders.get(identifier);
    if (!imageDecoder)
        return;

    connection->protectedConnection()->send(Messages::RemoteImageDecoderAVFManager::EncodedDataStatusChanged(identifier, imageDecoder->frameCount(), imageDecoder->size(), imageDecoder->hasTrack()), 0);
}

void RemoteImageDecoderAVFProxy::setExpectedContentSize(ImageDecoderIdentifier identifier, long long expectedContentSize)
{
    ASSERT(m_imageDecoders.contains(identifier));
    if (!m_imageDecoders.contains(identifier))
        return;

    RefPtr { m_imageDecoders.get(identifier) }->setExpectedContentSize(expectedContentSize);
}

void RemoteImageDecoderAVFProxy::setData(ImageDecoderIdentifier identifier, const IPC::SharedBufferReference& data, bool allDataReceived, CompletionHandler<void(size_t frameCount, const IntSize& size, bool hasTrack, std::optional<Vector<ImageDecoder::FrameInfo>>&&)>&& completionHandler)
{
    ASSERT(m_imageDecoders.contains(identifier));
    if (!m_imageDecoders.contains(identifier)) {
        completionHandler(0, { }, false, std::nullopt);
        return;
    }

    RefPtr imageDecoder = m_imageDecoders.get(identifier);
    imageDecoder->setData(data.isNull() ? SharedBuffer::create() : data.unsafeBuffer().releaseNonNull(), allDataReceived);

    auto frameCount = imageDecoder->frameCount();

    std::optional<Vector<ImageDecoder::FrameInfo>> frameInfos;
    if (frameCount)
        frameInfos = imageDecoder->frameInfos();

    completionHandler(frameCount, imageDecoder->size(), imageDecoder->hasTrack(), WTFMove(frameInfos));
}

void RemoteImageDecoderAVFProxy::createFrameImageAtIndex(ImageDecoderIdentifier identifier, size_t index, CompletionHandler<void(std::optional<WebCore::ShareableBitmap::Handle>&&)>&& completionHandler)
{
    ASSERT(m_imageDecoders.contains(identifier));

    std::optional<ShareableBitmap::Handle> imageHandle;

    auto invokeCallbackAtScopeExit = makeScopeExit([&] {
        completionHandler(WTFMove(imageHandle));
    });

    RefPtr imageDecoder = m_imageDecoders.get(identifier);
    if (!imageDecoder)
        return;

    auto nativeImage = NativeImage::createTransient(imageDecoder->createFrameImageAtIndex(index));
    if (!nativeImage)
        return;
    bool isOpaque = false;
    auto imageSize = nativeImage->size();
    auto bitmap = ShareableBitmap::create({ imageSize, nativeImage->colorSpace(), isOpaque });
    if (!bitmap)
        return;
    auto context = bitmap->createGraphicsContext();
    if (!context)
        return;

    FloatRect imageRect { { }, imageSize };
    context->drawNativeImage(*nativeImage, imageRect, imageRect, { CompositeOperator::Copy });
    imageHandle = bitmap->createHandle();
}

void RemoteImageDecoderAVFProxy::clearFrameBufferCache(ImageDecoderIdentifier identifier, size_t index)
{
    ASSERT(m_imageDecoders.contains(identifier));
    if (RefPtr imageDecoder = m_imageDecoders.get(identifier))
        imageDecoder->clearFrameBufferCache(std::min(index, imageDecoder->frameCount() - 1));
}

bool RemoteImageDecoderAVFProxy::allowsExitUnderMemoryPressure() const
{
    return m_imageDecoders.isEmpty();
}

std::optional<SharedPreferencesForWebProcess> RemoteImageDecoderAVFProxy::sharedPreferencesForWebProcess() const
{
    if (RefPtr connectionToWebProcess = m_connectionToWebProcess.get())
        return connectionToWebProcess->sharedPreferencesForWebProcess();

    return std::nullopt;
}

}

#endif
