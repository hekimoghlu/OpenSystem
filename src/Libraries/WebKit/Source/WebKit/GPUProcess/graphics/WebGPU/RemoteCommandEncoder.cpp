/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#include "RemoteCommandEncoder.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcess.h"
#include "Logging.h"
#include "RemoteCommandBuffer.h"
#include "RemoteCommandEncoderMessages.h"
#include "RemoteComputePassEncoder.h"
#include "RemoteRenderPassEncoder.h"
#include "StreamServerConnection.h"
#include "WebGPUComputePassDescriptor.h"
#include "WebGPUObjectHeap.h"
#include <WebCore/WebGPUBuffer.h>
#include <WebCore/WebGPUCommandEncoder.h>
#include <wtf/TZoneMallocInlines.h>

#define MESSAGE_CHECK(assertion) MESSAGE_CHECK_OPTIONAL_CONNECTION_BASE(assertion, connection())

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteCommandEncoder);

RemoteCommandEncoder::RemoteCommandEncoder(GPUConnectionToWebProcess& gpuConnectionToWebProcess, RemoteGPU& gpu, WebCore::WebGPU::CommandEncoder& commandEncoder, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, WebGPUIdentifier identifier)
    : m_backing(commandEncoder)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_identifier(identifier)
    , m_gpuConnectionToWebProcess(gpuConnectionToWebProcess)
    , m_gpu(gpu)
{
    Ref { m_streamConnection }->startReceivingMessages(*this, Messages::RemoteCommandEncoder::messageReceiverName(), m_identifier.toUInt64());
}

RemoteCommandEncoder::~RemoteCommandEncoder() = default;

RefPtr<IPC::Connection> RemoteCommandEncoder::connection() const
{
    RefPtr connection = m_gpuConnectionToWebProcess.get();
    if (!connection)
        return nullptr;
    return &connection->connection();
}

void RemoteCommandEncoder::destruct()
{
    protectedObjectHeap()->removeObject(m_identifier);
}

void RemoteCommandEncoder::stopListeningForIPC()
{
    Ref { m_streamConnection }->stopReceivingMessages(Messages::RemoteCommandEncoder::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteCommandEncoder::beginRenderPass(const WebGPU::RenderPassDescriptor& descriptor, WebGPUIdentifier identifier)
{
    Ref objectHeap = m_objectHeap.get();
    auto convertedDescriptor = objectHeap->convertFromBacking(descriptor);
    MESSAGE_CHECK(convertedDescriptor);

    auto renderPassEncoder = protectedBacking()->beginRenderPass(*convertedDescriptor);
    MESSAGE_CHECK(renderPassEncoder);
    auto remoteRenderPassEncoder = RemoteRenderPassEncoder::create(*renderPassEncoder, objectHeap, m_streamConnection.copyRef(), protectedGPU(), identifier);
    objectHeap->addObject(identifier, remoteRenderPassEncoder);
}

void RemoteCommandEncoder::beginComputePass(const std::optional<WebGPU::ComputePassDescriptor>& descriptor, WebGPUIdentifier identifier)
{
    Ref objectHeap = m_objectHeap.get();
    std::optional<WebCore::WebGPU::ComputePassDescriptor> convertedDescriptor;
    if (descriptor) {
        auto resultDescriptor = objectHeap->convertFromBacking(*descriptor);
        MESSAGE_CHECK(resultDescriptor);
        convertedDescriptor = WTFMove(resultDescriptor);
    }

    auto computePassEncoder = protectedBacking()->beginComputePass(convertedDescriptor);
    MESSAGE_CHECK(computePassEncoder);
    auto computeRenderPassEncoder = RemoteComputePassEncoder::create(*computePassEncoder, objectHeap, m_streamConnection.copyRef(), protectedGPU(), identifier);
    objectHeap->addObject(identifier, computeRenderPassEncoder);
}

void RemoteCommandEncoder::copyBufferToBuffer(
    WebGPUIdentifier source,
    WebCore::WebGPU::Size64 sourceOffset,
    WebGPUIdentifier destination,
    WebCore::WebGPU::Size64 destinationOffset,
    WebCore::WebGPU::Size64 size)
{
    Ref objectHeap = m_objectHeap.get();
    auto convertedSource = objectHeap->convertBufferFromBacking(source);
    ASSERT(convertedSource);
    auto convertedDestination = objectHeap->convertBufferFromBacking(destination);
    ASSERT(convertedDestination);
    if (!convertedSource || !convertedDestination)
        return;

    protectedBacking()->copyBufferToBuffer(*convertedSource, sourceOffset, *convertedDestination, destinationOffset, size);
}

void RemoteCommandEncoder::copyBufferToTexture(
    const WebGPU::ImageCopyBuffer& source,
    const WebGPU::ImageCopyTexture& destination,
    const WebGPU::Extent3D& copySize)
{
    auto convertedSource = m_objectHeap->convertFromBacking(source);
    ASSERT(convertedSource);
    auto convertedDestination = m_objectHeap->convertFromBacking(destination);
    ASSERT(convertedDestination);
    auto convertedCopySize = m_objectHeap->convertFromBacking(copySize);
    ASSERT(convertedCopySize);
    if (!convertedSource || !convertedDestination || !convertedCopySize)
        return;

    protectedBacking()->copyBufferToTexture(*convertedSource, *convertedDestination, *convertedCopySize);
}

void RemoteCommandEncoder::copyTextureToBuffer(
    const WebGPU::ImageCopyTexture& source,
    const WebGPU::ImageCopyBuffer& destination,
    const WebGPU::Extent3D& copySize)
{
    auto convertedSource = m_objectHeap->convertFromBacking(source);
    ASSERT(convertedSource);
    auto convertedDestination = m_objectHeap->convertFromBacking(destination);
    ASSERT(convertedDestination);
    auto convertedCopySize = m_objectHeap->convertFromBacking(copySize);
    ASSERT(convertedCopySize);
    if (!convertedSource || !convertedDestination || !convertedCopySize)
        return;

    protectedBacking()->copyTextureToBuffer(*convertedSource, *convertedDestination, *convertedCopySize);
}

void RemoteCommandEncoder::copyTextureToTexture(
    const WebGPU::ImageCopyTexture& source,
    const WebGPU::ImageCopyTexture& destination,
    const WebGPU::Extent3D& copySize)
{
    auto convertedSource = m_objectHeap->convertFromBacking(source);
    ASSERT(convertedSource);
    auto convertedDestination = m_objectHeap->convertFromBacking(destination);
    ASSERT(convertedDestination);
    auto convertedCopySize = m_objectHeap->convertFromBacking(copySize);
    ASSERT(convertedCopySize);
    if (!convertedSource || !convertedDestination || !convertedCopySize)
        return;

    protectedBacking()->copyTextureToTexture(*convertedSource, *convertedDestination, *convertedCopySize);
}

void RemoteCommandEncoder::clearBuffer(
    WebGPUIdentifier buffer,
    WebCore::WebGPU::Size64 offset,
    std::optional<WebCore::WebGPU::Size64> size)
{
    auto convertedBuffer = protectedObjectHeap()->convertBufferFromBacking(buffer);
    ASSERT(convertedBuffer);
    if (!convertedBuffer)
        return;

    protectedBacking()->clearBuffer(*convertedBuffer, offset, size);
}

void RemoteCommandEncoder::pushDebugGroup(String&& groupLabel)
{
    protectedBacking()->pushDebugGroup(WTFMove(groupLabel));
}

void RemoteCommandEncoder::popDebugGroup()
{
    protectedBacking()->popDebugGroup();
}

void RemoteCommandEncoder::insertDebugMarker(String&& markerLabel)
{
    protectedBacking()->insertDebugMarker(WTFMove(markerLabel));
}

void RemoteCommandEncoder::writeTimestamp(WebGPUIdentifier querySet, WebCore::WebGPU::Size32 queryIndex)
{
    auto convertedQuerySet = protectedObjectHeap()->convertQuerySetFromBacking(querySet);
    ASSERT(convertedQuerySet);
    if (!convertedQuerySet)
        return;

    protectedBacking()->writeTimestamp(*convertedQuerySet, queryIndex);
}

void RemoteCommandEncoder::resolveQuerySet(
    WebGPUIdentifier querySet,
    WebCore::WebGPU::Size32 firstQuery,
    WebCore::WebGPU::Size32 queryCount,
    WebGPUIdentifier destination,
    WebCore::WebGPU::Size64 destinationOffset)
{
    Ref objectHeap = m_objectHeap.get();
    auto convertedQuerySet = objectHeap->convertQuerySetFromBacking(querySet);
    ASSERT(convertedQuerySet);
    auto convertedDestination = objectHeap->convertBufferFromBacking(destination);
    ASSERT(convertedDestination);
    if (!convertedQuerySet || !convertedDestination)
        return;

    protectedBacking()->resolveQuerySet(*convertedQuerySet, firstQuery, queryCount, *convertedDestination, destinationOffset);
}

void RemoteCommandEncoder::finish(const WebGPU::CommandBufferDescriptor& descriptor, WebGPUIdentifier identifier)
{
    Ref objectHeap = m_objectHeap.get();
    auto convertedDescriptor = objectHeap->convertFromBacking(descriptor);
    MESSAGE_CHECK(convertedDescriptor);

    auto commandBuffer = protectedBacking()->finish(*convertedDescriptor);
    MESSAGE_CHECK(commandBuffer);
    auto remoteCommandBuffer = RemoteCommandBuffer::create(*commandBuffer, objectHeap, m_streamConnection.copyRef(), protectedGPU(), identifier);
    objectHeap->addObject(identifier, remoteCommandBuffer);
}

void RemoteCommandEncoder::setLabel(String&& label)
{
    protectedBacking()->setLabel(WTFMove(label));
}

Ref<WebCore::WebGPU::CommandEncoder> RemoteCommandEncoder::protectedBacking()
{
    return m_backing;
}

} // namespace WebKit

#undef MESSAGE_CHECK

#endif // ENABLE(GPU_PROCESS)
