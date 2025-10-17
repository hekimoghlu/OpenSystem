/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#include "WebGPUCommandEncoderImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUBufferImpl.h"
#include "WebGPUCommandBufferImpl.h"
#include "WebGPUComputePassEncoderImpl.h"
#include "WebGPUConvertToBackingContext.h"
#include "WebGPUQuerySetImpl.h"
#include "WebGPURenderPassEncoderImpl.h"
#include "WebGPUTextureImpl.h"
#include "WebGPUTextureViewImpl.h"
#include <WebGPU/WebGPUExt.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CommandEncoderImpl);

CommandEncoderImpl::CommandEncoderImpl(WebGPUPtr<WGPUCommandEncoder>&& commandEncoder, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(commandEncoder))
    , m_convertToBackingContext(convertToBackingContext)
{
}

CommandEncoderImpl::~CommandEncoderImpl() = default;

RefPtr<RenderPassEncoder> CommandEncoderImpl::beginRenderPass(const RenderPassDescriptor& descriptor)
{
    auto label = descriptor.label.utf8();

    Vector<WGPURenderPassColorAttachment> colorAttachments;
    Ref convertToBackingContext = m_convertToBackingContext;
    for (const auto& colorAttachment : descriptor.colorAttachments) {
        if (colorAttachment) {
            colorAttachments.append(WGPURenderPassColorAttachment {
                .nextInChain = nullptr,
                .view = convertToBackingContext->convertToBacking(colorAttachment->protectedView().get()),
                .depthSlice = colorAttachment->depthSlice,
                .resolveTarget = colorAttachment->resolveTarget ? convertToBackingContext->convertToBacking(*colorAttachment->protectedResolveTarget()) : nullptr,
                .loadOp = convertToBackingContext->convertToBacking(colorAttachment->loadOp),
                .storeOp = convertToBackingContext->convertToBacking(colorAttachment->storeOp),
                .clearValue = colorAttachment->clearValue ? convertToBackingContext->convertToBacking(*colorAttachment->clearValue) : WGPUColor { 0, 0, 0, 0 },
            });
        } else
            colorAttachments.append(WGPURenderPassColorAttachment {
                .nextInChain = nullptr,
                .view = nullptr,
                .depthSlice = std::nullopt,
                .resolveTarget = nullptr,
                .loadOp = WGPULoadOp_Clear,
                .storeOp = WGPUStoreOp_Discard,
                .clearValue = { 0, 0, 0, 0 },
            });
    }

    std::optional<WGPURenderPassDepthStencilAttachment> depthStencilAttachment;
    if (descriptor.depthStencilAttachment) {
        depthStencilAttachment = WGPURenderPassDepthStencilAttachment {
            .view = convertToBackingContext->convertToBacking(descriptor.depthStencilAttachment->protectedView().get()),
            .depthLoadOp = descriptor.depthStencilAttachment->depthLoadOp ? convertToBackingContext->convertToBacking(*descriptor.depthStencilAttachment->depthLoadOp) : WGPULoadOp_Undefined,
            .depthStoreOp = descriptor.depthStencilAttachment->depthStoreOp ? convertToBackingContext->convertToBacking(*descriptor.depthStencilAttachment->depthStoreOp) : WGPUStoreOp_Undefined,
            .depthClearValue = descriptor.depthStencilAttachment->depthClearValue,
            .depthReadOnly = descriptor.depthStencilAttachment->depthReadOnly,
            .stencilLoadOp = descriptor.depthStencilAttachment->stencilLoadOp ? convertToBackingContext->convertToBacking(*descriptor.depthStencilAttachment->stencilLoadOp) : WGPULoadOp_Undefined,
            .stencilStoreOp = descriptor.depthStencilAttachment->stencilStoreOp ? convertToBackingContext->convertToBacking(*descriptor.depthStencilAttachment->stencilStoreOp) : WGPUStoreOp_Undefined,
            .stencilClearValue = descriptor.depthStencilAttachment->stencilClearValue,
            .stencilReadOnly = descriptor.depthStencilAttachment->stencilReadOnly,
        };
    }

    WGPURenderPassTimestampWrites timestampWrites {
        .querySet = descriptor.timestampWrites ? convertToBackingContext->convertToBacking(*descriptor.timestampWrites->protectedQuerySet()) : nullptr,
        .beginningOfPassWriteIndex = descriptor.timestampWrites ? descriptor.timestampWrites->beginningOfPassWriteIndex : 0,
        .endOfPassWriteIndex = descriptor.timestampWrites ? descriptor.timestampWrites->endOfPassWriteIndex : 0
    };

    WGPURenderPassDescriptorMaxDrawCount maxDrawCount {
        .chain = {
            nullptr,
            WGPUSType_RenderPassDescriptorMaxDrawCount,
        },
        .maxDrawCount = descriptor.maxDrawCount.value_or(0)
    };
    WGPURenderPassDescriptor backingDescriptor {
        .nextInChain = descriptor.maxDrawCount ? &maxDrawCount.chain : nullptr,
        .label = label.data(),
        .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
        .colorAttachments = colorAttachments.data(),
        .depthStencilAttachment = depthStencilAttachment ? &depthStencilAttachment.value() : nullptr,
        .occlusionQuerySet = descriptor.occlusionQuerySet ? convertToBackingContext->convertToBacking(*descriptor.protectedOcclusionQuerySet()) : nullptr,
        .timestampWrites = timestampWrites.querySet ? &timestampWrites : nullptr
    };

    return RenderPassEncoderImpl::create(adoptWebGPU(wgpuCommandEncoderBeginRenderPass(m_backing.get(), &backingDescriptor)), convertToBackingContext);
}

RefPtr<ComputePassEncoder> CommandEncoderImpl::beginComputePass(const std::optional<ComputePassDescriptor>& descriptor)
{
    CString label = descriptor ? descriptor->label.utf8() : CString("");

    WGPUComputePassTimestampWrites timestampWrites {
        .querySet = (descriptor && descriptor->timestampWrites && descriptor->timestampWrites->querySet) ? protectedConvertToBackingContext()->convertToBacking(*descriptor->timestampWrites->protectedQuerySet().get()) : nullptr,
        .beginningOfPassWriteIndex = (descriptor && descriptor->timestampWrites) ? descriptor->timestampWrites->beginningOfPassWriteIndex : 0,
        .endOfPassWriteIndex = (descriptor && descriptor->timestampWrites) ? descriptor->timestampWrites->endOfPassWriteIndex : 0
    };

    WGPUComputePassDescriptor backingDescriptor {
        .nextInChain = nullptr,
        .label = label.data(),
        .timestampWrites = timestampWrites.querySet ? &timestampWrites : nullptr
    };

    return ComputePassEncoderImpl::create(adoptWebGPU(wgpuCommandEncoderBeginComputePass(m_backing.get(), &backingDescriptor)), m_convertToBackingContext);
}

void CommandEncoderImpl::copyBufferToBuffer(
    const Buffer& source,
    Size64 sourceOffset,
    const Buffer& destination,
    Size64 destinationOffset,
    Size64 size)
{
    Ref convertToBackingContext = m_convertToBackingContext;
    wgpuCommandEncoderCopyBufferToBuffer(m_backing.get(), convertToBackingContext->convertToBacking(source), sourceOffset, convertToBackingContext->convertToBacking(destination), destinationOffset, size);
}

void CommandEncoderImpl::copyBufferToTexture(
    const ImageCopyBuffer& source,
    const ImageCopyTexture& destination,
    const Extent3D& copySize)
{
    Ref convertToBackingContext = m_convertToBackingContext;

    WGPUImageCopyBuffer backingSource {
        nullptr, {
            nullptr,
            source.offset,
            source.bytesPerRow.value_or(WGPU_COPY_STRIDE_UNDEFINED),
            source.rowsPerImage.value_or(WGPU_COPY_STRIDE_UNDEFINED),
        },
        convertToBackingContext->convertToBacking(source.protectedBuffer().get()),
    };

    WGPUImageCopyTexture backingDestination {
        nullptr,
        convertToBackingContext->convertToBacking(destination.protectedTexture().get()),
        destination.mipLevel,
        destination.origin ? convertToBackingContext->convertToBacking(*destination.origin) : WGPUOrigin3D { 0, 0, 0 },
        convertToBackingContext->convertToBacking(destination.aspect),
    };

    WGPUExtent3D backingCopySize = convertToBackingContext->convertToBacking(copySize);

    wgpuCommandEncoderCopyBufferToTexture(m_backing.get(), &backingSource, &backingDestination, &backingCopySize);
}

void CommandEncoderImpl::copyTextureToBuffer(
    const ImageCopyTexture& source,
    const ImageCopyBuffer& destination,
    const Extent3D& copySize)
{
    Ref convertToBackingContext = m_convertToBackingContext;

    WGPUImageCopyTexture backingSource {
        nullptr,
        convertToBackingContext->convertToBacking(source.protectedTexture().get()),
        source.mipLevel,
        source.origin ? convertToBackingContext->convertToBacking(*source.origin) : WGPUOrigin3D { 0, 0, 0 },
        convertToBackingContext->convertToBacking(source.aspect),
    };

    WGPUImageCopyBuffer backingDestination {
        nullptr, {
            nullptr,
            destination.offset,
            destination.bytesPerRow.value_or(WGPU_COPY_STRIDE_UNDEFINED),
            destination.rowsPerImage.value_or(WGPU_COPY_STRIDE_UNDEFINED),
        },
        convertToBackingContext->convertToBacking(destination.protectedBuffer()),
    };

    WGPUExtent3D backingCopySize = convertToBackingContext->convertToBacking(copySize);

    wgpuCommandEncoderCopyTextureToBuffer(m_backing.get(), &backingSource, &backingDestination, &backingCopySize);
}

void CommandEncoderImpl::copyTextureToTexture(
    const ImageCopyTexture& source,
    const ImageCopyTexture& destination,
    const Extent3D& copySize)
{
    Ref convertToBackingContext = m_convertToBackingContext;

    WGPUImageCopyTexture backingSource {
        nullptr,
        convertToBackingContext->convertToBacking(source.protectedTexture().get()),
        source.mipLevel,
        source.origin ? convertToBackingContext->convertToBacking(*source.origin) : WGPUOrigin3D { 0, 0, 0 },
        convertToBackingContext->convertToBacking(source.aspect),
    };

    WGPUImageCopyTexture backingDestination {
        nullptr,
        convertToBackingContext->convertToBacking(destination.protectedTexture().get()),
        destination.mipLevel,
        destination.origin ? convertToBackingContext->convertToBacking(*destination.origin) : WGPUOrigin3D { 0, 0, 0 },
        convertToBackingContext->convertToBacking(destination.aspect),
    };

    WGPUExtent3D backingCopySize = convertToBackingContext->convertToBacking(copySize);

    wgpuCommandEncoderCopyTextureToTexture(m_backing.get(), &backingSource, &backingDestination, &backingCopySize);
}

void CommandEncoderImpl::clearBuffer(
    const Buffer& buffer,
    Size64 offset,
    std::optional<Size64> size)
{
    wgpuCommandEncoderClearBuffer(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(buffer), offset, size.value_or(WGPU_WHOLE_SIZE));
}

void CommandEncoderImpl::pushDebugGroup(String&& groupLabel)
{
    wgpuCommandEncoderPushDebugGroup(m_backing.get(), groupLabel.utf8().data());
}

void CommandEncoderImpl::popDebugGroup()
{
    wgpuCommandEncoderPopDebugGroup(m_backing.get());
}

void CommandEncoderImpl::insertDebugMarker(String&& markerLabel)
{
    wgpuCommandEncoderInsertDebugMarker(m_backing.get(), markerLabel.utf8().data());
}

void CommandEncoderImpl::writeTimestamp(const QuerySet& querySet, Size32 queryIndex)
{
    wgpuCommandEncoderWriteTimestamp(m_backing.get(), protectedConvertToBackingContext()->convertToBacking(querySet), queryIndex);
}

void CommandEncoderImpl::resolveQuerySet(
    const QuerySet& querySet,
    Size32 firstQuery,
    Size32 queryCount,
    const Buffer& destination,
    Size64 destinationOffset)
{
    Ref convertToBackingContext = m_convertToBackingContext;
    wgpuCommandEncoderResolveQuerySet(m_backing.get(), convertToBackingContext->convertToBacking(querySet), firstQuery, queryCount, convertToBackingContext->convertToBacking(destination), destinationOffset);
}

RefPtr<CommandBuffer> CommandEncoderImpl::finish(const CommandBufferDescriptor& descriptor)
{
    auto label = descriptor.label.utf8();

    WGPUCommandBufferDescriptor backingDescriptor {
        nullptr,
        label.data(),
    };

    return CommandBufferImpl::create(adoptWebGPU(wgpuCommandEncoderFinish(m_backing.get(), &backingDescriptor)), m_convertToBackingContext);
}

void CommandEncoderImpl::setLabelInternal(const String& label)
{
    wgpuCommandEncoderSetLabel(m_backing.get(), label.utf8().data());
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
