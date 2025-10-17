/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#include "GPUCommandEncoder.h"

#include "GPUBuffer.h"
#include "GPUCommandBuffer.h"
#include "GPUQuerySet.h"
#include "WebGPUDevice.h"

namespace WebCore {

GPUCommandEncoder::GPUCommandEncoder(Ref<WebGPU::CommandEncoder>&& backing, WebGPU::Device& device)
    : m_backing(WTFMove(backing))
    , m_device(&device)
{
}

String GPUCommandEncoder::label() const
{
    return m_backing->label();
}

void GPUCommandEncoder::setLabel(String&& label)
{
    m_backing->setLabel(WTFMove(label));
}

ExceptionOr<Ref<GPURenderPassEncoder>> GPUCommandEncoder::beginRenderPass(const GPURenderPassDescriptor& renderPassDescriptor)
{
    RefPtr encoder = m_backing->beginRenderPass(renderPassDescriptor.convertToBacking());
    if (!encoder || !m_device.get())
        return Exception { ExceptionCode::InvalidStateError, "GPUCommandEncoder.beginRenderPass: Unable to begin render pass."_s };
    return GPURenderPassEncoder::create(encoder.releaseNonNull(), *m_device.get());
}

ExceptionOr<Ref<GPUComputePassEncoder>> GPUCommandEncoder::beginComputePass(const std::optional<GPUComputePassDescriptor>& computePassDescriptor)
{
    RefPtr computePass = m_backing->beginComputePass(computePassDescriptor ? std::optional { computePassDescriptor->convertToBacking() } : std::nullopt);
    if (!computePass || !m_device.get())
        return Exception { ExceptionCode::InvalidStateError, "GPUCommandEncoder.beginComputePass: Unable to begin compute pass."_s };
    return GPUComputePassEncoder::create(computePass.releaseNonNull(), *m_device.get());
}

void GPUCommandEncoder::copyBufferToBuffer(
    const GPUBuffer& source,
    GPUSize64 sourceOffset,
    const GPUBuffer& destination,
    GPUSize64 destinationOffset,
    GPUSize64 size)
{
    m_backing->copyBufferToBuffer(source.backing(), sourceOffset, destination.backing(), destinationOffset, size);
}

void GPUCommandEncoder::copyBufferToTexture(
    const GPUImageCopyBuffer& source,
    const GPUImageCopyTexture& destination,
    const GPUExtent3D& copySize)
{
    m_backing->copyBufferToTexture(source.convertToBacking(), destination.convertToBacking(), convertToBacking(copySize));
}

void GPUCommandEncoder::copyTextureToBuffer(
    const GPUImageCopyTexture& source,
    const GPUImageCopyBuffer& destination,
    const GPUExtent3D& copySize)
{
    m_backing->copyTextureToBuffer(source.convertToBacking(), destination.convertToBacking(), convertToBacking(copySize));
}

void GPUCommandEncoder::copyTextureToTexture(
    const GPUImageCopyTexture& source,
    const GPUImageCopyTexture& destination,
    const GPUExtent3D& copySize)
{
    m_backing->copyTextureToTexture(source.convertToBacking(), destination.convertToBacking(), convertToBacking(copySize));
}


void GPUCommandEncoder::clearBuffer(
    const GPUBuffer& buffer,
    std::optional<GPUSize64> offset,
    std::optional<GPUSize64> size)
{
    m_backing->clearBuffer(buffer.backing(), offset.value_or(0), size);
}

void GPUCommandEncoder::pushDebugGroup(String&& groupLabel)
{
    m_backing->pushDebugGroup(WTFMove(groupLabel));
}

void GPUCommandEncoder::popDebugGroup()
{
    m_backing->popDebugGroup();
}

void GPUCommandEncoder::insertDebugMarker(String&& markerLabel)
{
    m_backing->insertDebugMarker(WTFMove(markerLabel));
}

void GPUCommandEncoder::writeTimestamp(const GPUQuerySet& querySet, GPUSize32 queryIndex)
{
    m_backing->writeTimestamp(querySet.backing(), queryIndex);
}

void GPUCommandEncoder::resolveQuerySet(
    const GPUQuerySet& querySet,
    GPUSize32 firstQuery,
    GPUSize32 queryCount,
    const GPUBuffer& destination,
    GPUSize64 destinationOffset)
{
    m_backing->resolveQuerySet(querySet.backing(), firstQuery, queryCount, destination.backing(), destinationOffset);
}

static WebGPU::CommandBufferDescriptor convertToBacking(const std::optional<GPUCommandBufferDescriptor>& commandBufferDescriptor)
{
    if (!commandBufferDescriptor)
        return { };

    return commandBufferDescriptor->convertToBacking();
}

ExceptionOr<Ref<GPUCommandBuffer>> GPUCommandEncoder::finish(const std::optional<GPUCommandBufferDescriptor>& commandBufferDescriptor)
{
    RefPtr buffer = m_backing->finish(convertToBacking(commandBufferDescriptor));
    if (!buffer)
        return Exception { ExceptionCode::InvalidStateError, "GPUCommandEncoder.finish: Unable to finish."_s };
    return GPUCommandBuffer::create(buffer.releaseNonNull(), *this);
}

void GPUCommandEncoder::setBacking(WebGPU::CommandEncoder& newBacking)
{
    m_backing = newBacking;
}

}
