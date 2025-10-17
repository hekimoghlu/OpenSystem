/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#include "RemoteQueueProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteQueueMessages.h"
#include "WebGPUConvertToBackingContext.h"
#include "WebProcess.h"
#include <WebCore/NativeImage.h>
#include <WebCore/WebCodecsVideoFrame.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteQueueProxy);

RemoteQueueProxy::RemoteQueueProxy(RemoteAdapterProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_parent(parent)
{
#if ENABLE(VIDEO) && PLATFORM(COCOA) && ENABLE(WEB_CODECS)
    RefPtr<RemoteVideoFrameObjectHeapProxy> videoFrameObjectHeapProxy;
    callOnMainRunLoopAndWait([&videoFrameObjectHeapProxy] {
        videoFrameObjectHeapProxy = WebProcess::singleton().ensureProtectedGPUProcessConnection()->protectedVideoFrameObjectHeapProxy();
    });

    m_videoFrameObjectHeapProxy = videoFrameObjectHeapProxy;
#endif
}

RemoteQueueProxy::~RemoteQueueProxy()
{
    auto sendResult = send(Messages::RemoteQueue::Destruct());
    UNUSED_VARIABLE(sendResult);
}

void RemoteQueueProxy::submit(Vector<Ref<WebCore::WebGPU::CommandBuffer>>&& commandBuffers)
{
    auto convertedCommandBuffers = WTF::compactMap(commandBuffers, [&](auto& commandBuffer) -> std::optional<WebGPUIdentifier> {
        auto convertedCommandBuffer = protectedConvertToBackingContext()->convertToBacking(commandBuffer);
        return convertedCommandBuffer;
    });

    auto sendResult = send(Messages::RemoteQueue::Submit(convertedCommandBuffers));
    UNUSED_VARIABLE(sendResult);
}

void RemoteQueueProxy::onSubmittedWorkDone(CompletionHandler<void()>&& callback)
{
    auto sendResult = sendWithAsyncReply(Messages::RemoteQueue::OnSubmittedWorkDone(), [callback = WTFMove(callback)]() mutable {
        callback();
    });
    UNUSED_PARAM(sendResult);
}

void RemoteQueueProxy::writeBuffer(
    const WebCore::WebGPU::Buffer& buffer,
    WebCore::WebGPU::Size64 bufferOffset,
    std::span<const uint8_t> source,
    WebCore::WebGPU::Size64 dataOffset,
    std::optional<WebCore::WebGPU::Size64> size)
{
    auto convertedBuffer = protectedConvertToBackingContext()->convertToBacking(buffer);

    auto sharedMemory = WebCore::SharedMemory::copySpan(source.subspan(dataOffset, static_cast<size_t>(size.value_or(source.size() - dataOffset))));
    std::optional<WebCore::SharedMemoryHandle> handle;
    if (sharedMemory)
        handle = sharedMemory->createHandle(WebCore::SharedMemory::Protection::ReadOnly);
    auto sendResult = sendWithAsyncReply(Messages::RemoteQueue::WriteBuffer(convertedBuffer, bufferOffset, WTFMove(handle)), [sharedMemory = sharedMemory.copyRef(), handleHasValue = handle.has_value()](auto) mutable {
        RELEASE_ASSERT(sharedMemory.get() || !handleHasValue);
    });
    UNUSED_VARIABLE(sendResult);
}

void RemoteQueueProxy::writeTexture(
    const WebCore::WebGPU::ImageCopyTexture& destination,
    std::span<const uint8_t> source,
    const WebCore::WebGPU::ImageDataLayout& dataLayout,
    const WebCore::WebGPU::Extent3D& size)
{
    Ref convertToBackingContext = m_convertToBackingContext;
    auto convertedDestination = convertToBackingContext->convertToBacking(destination);
    ASSERT(convertedDestination);
    auto convertedDataLayout = convertToBackingContext->convertToBacking(dataLayout);
    ASSERT(convertedDataLayout);
    auto convertedSize = convertToBackingContext->convertToBacking(size);
    ASSERT(convertedSize);
    if (!convertedDestination || !convertedDataLayout || !convertedSize)
        return;

    auto sharedMemory = WebCore::SharedMemory::copySpan(source);
    std::optional<WebCore::SharedMemoryHandle> handle;
    if (sharedMemory)
        handle = sharedMemory->createHandle(WebCore::SharedMemory::Protection::ReadOnly);
    auto sendResult = sendWithAsyncReply(Messages::RemoteQueue::WriteTexture(*convertedDestination, WTFMove(handle), *convertedDataLayout, *convertedSize), [sharedMemory = sharedMemory.copyRef(), handleHasValue = handle.has_value()](auto) mutable {
        RELEASE_ASSERT(sharedMemory.get() || !handleHasValue);
    });
    UNUSED_VARIABLE(sendResult);
}

void RemoteQueueProxy::writeBufferNoCopy(
    const WebCore::WebGPU::Buffer&,
    WebCore::WebGPU::Size64,
    std::span<uint8_t>,
    WebCore::WebGPU::Size64,
    std::optional<WebCore::WebGPU::Size64>)
{
    RELEASE_ASSERT_NOT_REACHED();
}

void RemoteQueueProxy::writeTexture(
    const WebCore::WebGPU::ImageCopyTexture&,
    std::span<uint8_t>,
    const WebCore::WebGPU::ImageDataLayout&,
    const WebCore::WebGPU::Extent3D&)
{
    RELEASE_ASSERT_NOT_REACHED();
}

void RemoteQueueProxy::copyExternalImageToTexture(
    const WebCore::WebGPU::ImageCopyExternalImage& source,
    const WebCore::WebGPU::ImageCopyTextureTagged& destination,
    const WebCore::WebGPU::Extent3D& copySize)
{
    Ref convertToBackingContext = m_convertToBackingContext;
    auto convertedSource = convertToBackingContext->convertToBacking(source);
    ASSERT(convertedSource);
    auto convertedDestination = convertToBackingContext->convertToBacking(destination);
    ASSERT(convertedDestination);
    auto convertedCopySize = convertToBackingContext->convertToBacking(copySize);
    ASSERT(convertedCopySize);
    if (!convertedSource || !convertedDestination || !convertedCopySize)
        return;

    auto sendResult = send(Messages::RemoteQueue::CopyExternalImageToTexture(*convertedSource, *convertedDestination, *convertedCopySize));
    UNUSED_VARIABLE(sendResult);
}

void RemoteQueueProxy::setLabelInternal(const String& label)
{
    auto sendResult = send(Messages::RemoteQueue::SetLabel(label));
    UNUSED_VARIABLE(sendResult);
}

Ref<ConvertToBackingContext> RemoteQueueProxy::protectedConvertToBackingContext() const
{
    return m_convertToBackingContext;
}

RefPtr<WebCore::NativeImage> RemoteQueueProxy::getNativeImage(WebCore::VideoFrame& videoFrame)
{
    RefPtr<WebCore::NativeImage> nativeImage;
#if ENABLE(VIDEO) && PLATFORM(COCOA) && ENABLE(WEB_CODECS)
    callOnMainRunLoopAndWait([&nativeImage, videoFrame = Ref { videoFrame }, videoFrameHeap = protectedVideoFrameObjectHeapProxy()] {
        nativeImage = videoFrameHeap->getNativeImage(videoFrame);
    });
#endif
    return nativeImage;
}

#if ENABLE(VIDEO)
RefPtr<RemoteVideoFrameObjectHeapProxy> RemoteQueueProxy::protectedVideoFrameObjectHeapProxy() const
{
    return m_videoFrameObjectHeapProxy;
}
#endif


} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
