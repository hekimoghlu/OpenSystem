/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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
#include "WebGPUQueueImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUBufferImpl.h"
#include "WebGPUCommandBufferImpl.h"
#include "WebGPUConvertToBackingContext.h"
#include "WebGPUTextureImpl.h"
#include <WebGPU/WebGPUExt.h>
#include <wtf/BlockPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(QueueImpl);

QueueImpl::QueueImpl(WebGPUPtr<WGPUQueue>&& queue, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(queue))
    , m_convertToBackingContext(convertToBackingContext)
{
}

QueueImpl::~QueueImpl() = default;

void QueueImpl::submit(Vector<Ref<WebGPU::CommandBuffer>>&& commandBuffers)
{
    auto backingCommandBuffers = commandBuffers.map([&](auto commandBuffer) {
        return Ref { m_convertToBackingContext }->convertToBacking(commandBuffer);
    });

    wgpuQueueSubmit(m_backing.get(), backingCommandBuffers.size(), backingCommandBuffers.data());
}

static void onSubmittedWorkDoneCallback(WGPUQueueWorkDoneStatus status, void* userdata)
{
    auto block = reinterpret_cast<void(^)(WGPUQueueWorkDoneStatus)>(userdata);
    block(status);
    Block_release(block); // Block_release is matched with Block_copy below in QueueImpl::submit().
}

void QueueImpl::onSubmittedWorkDone(CompletionHandler<void()>&& callback)
{
    auto blockPtr = makeBlockPtr([callback = WTFMove(callback)](WGPUQueueWorkDoneStatus) mutable {
        callback();
    });
    wgpuQueueOnSubmittedWorkDone(m_backing.get(), &onSubmittedWorkDoneCallback, Block_copy(blockPtr.get())); // Block_copy is matched with Block_release above in onSubmittedWorkDoneCallback().
}

void QueueImpl::writeBuffer(
    const Buffer&,
    Size64,
    std::span<const uint8_t>,
    Size64,
    std::optional<Size64>)
{
    RELEASE_ASSERT_NOT_REACHED();
}

void QueueImpl::writeTexture(
    const ImageCopyTexture&,
    std::span<const uint8_t>,
    const ImageDataLayout&,
    const Extent3D&)
{
    RELEASE_ASSERT_NOT_REACHED();
}

void QueueImpl::writeBufferNoCopy(
    const Buffer& buffer,
    Size64 bufferOffset,
    std::span<uint8_t> source,
    Size64 dataOffset,
    std::optional<Size64> size)
{
    wgpuQueueWriteBuffer(m_backing.get(), Ref { m_convertToBackingContext }->convertToBacking(buffer), bufferOffset, source.subspan(dataOffset, size.value_or(source.size() - dataOffset)));
}

void QueueImpl::writeTexture(
    const ImageCopyTexture& destination,
    std::span<uint8_t> source,
    const ImageDataLayout& dataLayout,
    const Extent3D& size)
{
    Ref convertToBackingContext = m_convertToBackingContext;

    WGPUImageCopyTexture backingDestination {
        .nextInChain = nullptr,
        .texture = convertToBackingContext->convertToBacking(destination.protectedTexture().get()),
        .mipLevel = destination.mipLevel,
        .origin = destination.origin ? convertToBackingContext->convertToBacking(*destination.origin) : WGPUOrigin3D { 0, 0, 0 },
        .aspect = convertToBackingContext->convertToBacking(destination.aspect),
    };

    WGPUTextureDataLayout backingDataLayout {
        .nextInChain = nullptr,
        .offset = dataLayout.offset,
        .bytesPerRow = dataLayout.bytesPerRow.value_or(WGPU_COPY_STRIDE_UNDEFINED),
        .rowsPerImage = dataLayout.rowsPerImage.value_or(WGPU_COPY_STRIDE_UNDEFINED),
    };

    WGPUExtent3D backingSize = convertToBackingContext->convertToBacking(size);

    wgpuQueueWriteTexture(m_backing.get(), &backingDestination, source, &backingDataLayout, &backingSize);
}

void QueueImpl::copyExternalImageToTexture(
    const ImageCopyExternalImage& source,
    const ImageCopyTextureTagged& destination,
    const Extent3D& copySize)
{
    UNUSED_PARAM(source);
    UNUSED_PARAM(destination);
    UNUSED_PARAM(copySize);
}

void QueueImpl::setLabelInternal(const String& label)
{
    wgpuQueueSetLabel(m_backing.get(), label.utf8().data());
}

RefPtr<WebCore::NativeImage> QueueImpl::getNativeImage(WebCore::VideoFrame&)
{
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
