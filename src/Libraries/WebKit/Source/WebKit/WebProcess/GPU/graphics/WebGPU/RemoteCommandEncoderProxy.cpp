/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
#include "RemoteCommandEncoderProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteCommandBufferProxy.h"
#include "RemoteCommandEncoderMessages.h"
#include "RemoteComputePassEncoderProxy.h"
#include "RemoteRenderPassEncoderProxy.h"
#include "WebGPUConvertToBackingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteCommandEncoderProxy);

RemoteCommandEncoderProxy::RemoteCommandEncoderProxy(RemoteGPUProxy& root, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_root(root)
{
}

RemoteCommandEncoderProxy::~RemoteCommandEncoderProxy()
{
    auto sendResult = send(Messages::RemoteCommandEncoder::Destruct());
    UNUSED_VARIABLE(sendResult);
}

RefPtr<WebCore::WebGPU::RenderPassEncoder> RemoteCommandEncoderProxy::beginRenderPass(const WebCore::WebGPU::RenderPassDescriptor& descriptor)
{
    Ref convertToBackingContext = m_convertToBackingContext;
    auto convertedDescriptor = convertToBackingContext->convertToBacking(descriptor);

    if (!convertedDescriptor)
        return nullptr;

    auto identifier = WebGPUIdentifier::generate();
    auto sendResult = send(Messages::RemoteCommandEncoder::BeginRenderPass(*convertedDescriptor, identifier));
    if (sendResult != IPC::Error::NoError)
        return nullptr;

    auto result = RemoteRenderPassEncoderProxy::create(*this, convertToBackingContext, identifier);
    if (convertedDescriptor)
        result->setLabel(WTFMove(convertedDescriptor->label));
    return result;
}

RefPtr<WebCore::WebGPU::ComputePassEncoder> RemoteCommandEncoderProxy::beginComputePass(const std::optional<WebCore::WebGPU::ComputePassDescriptor>& descriptor)
{
    Ref convertToBackingContext = m_convertToBackingContext;
    std::optional<WebKit::WebGPU::ComputePassDescriptor> convertedDescriptor;

    if (descriptor) {
        convertedDescriptor = convertToBackingContext->convertToBacking(*descriptor);
        if (!convertedDescriptor)
            return nullptr;
    }

    auto identifier = WebGPUIdentifier::generate();
    auto sendResult = send(Messages::RemoteCommandEncoder::BeginComputePass(convertedDescriptor, identifier));
    if (sendResult != IPC::Error::NoError)
        return nullptr;

    auto result = RemoteComputePassEncoderProxy::create(*this, convertToBackingContext, identifier);
    if (convertedDescriptor)
        result->setLabel(WTFMove(convertedDescriptor->label));
    return result;
}

void RemoteCommandEncoderProxy::copyBufferToBuffer(
    const WebCore::WebGPU::Buffer& source,
    WebCore::WebGPU::Size64 sourceOffset,
    const WebCore::WebGPU::Buffer& destination,
    WebCore::WebGPU::Size64 destinationOffset,
    WebCore::WebGPU::Size64 size)
{
    Ref convertToBackingContext = m_convertToBackingContext;
    auto convertedSource = convertToBackingContext->convertToBacking(source);
    auto convertedDestination = convertToBackingContext->convertToBacking(destination);

    auto sendResult = send(Messages::RemoteCommandEncoder::CopyBufferToBuffer(convertedSource, sourceOffset, convertedDestination, destinationOffset, size));
    UNUSED_VARIABLE(sendResult);
}

void RemoteCommandEncoderProxy::copyBufferToTexture(
    const WebCore::WebGPU::ImageCopyBuffer& source,
    const WebCore::WebGPU::ImageCopyTexture& destination,
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

    auto sendResult = send(Messages::RemoteCommandEncoder::CopyBufferToTexture(*convertedSource, *convertedDestination, *convertedCopySize));
    UNUSED_VARIABLE(sendResult);
}

void RemoteCommandEncoderProxy::copyTextureToBuffer(
    const WebCore::WebGPU::ImageCopyTexture& source,
    const WebCore::WebGPU::ImageCopyBuffer& destination,
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

    auto sendResult = send(Messages::RemoteCommandEncoder::CopyTextureToBuffer(*convertedSource, *convertedDestination, *convertedCopySize));
    UNUSED_VARIABLE(sendResult);
}

void RemoteCommandEncoderProxy::copyTextureToTexture(
    const WebCore::WebGPU::ImageCopyTexture& source,
    const WebCore::WebGPU::ImageCopyTexture& destination,
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

    auto sendResult = send(Messages::RemoteCommandEncoder::CopyTextureToTexture(*convertedSource, *convertedDestination, *convertedCopySize));
    UNUSED_VARIABLE(sendResult);
}

void RemoteCommandEncoderProxy::clearBuffer(
    const WebCore::WebGPU::Buffer& buffer,
    WebCore::WebGPU::Size64 offset,
    std::optional<WebCore::WebGPU::Size64> size)
{
    auto convertedBuffer = protectedConvertToBackingContext()->convertToBacking(buffer);

    auto sendResult = send(Messages::RemoteCommandEncoder::ClearBuffer(convertedBuffer, offset, size));
    UNUSED_VARIABLE(sendResult);
}

void RemoteCommandEncoderProxy::pushDebugGroup(String&& groupLabel)
{
    auto sendResult = send(Messages::RemoteCommandEncoder::PushDebugGroup(WTFMove(groupLabel)));
    UNUSED_VARIABLE(sendResult);
}

void RemoteCommandEncoderProxy::popDebugGroup()
{
    auto sendResult = send(Messages::RemoteCommandEncoder::PopDebugGroup());
    UNUSED_VARIABLE(sendResult);
}

void RemoteCommandEncoderProxy::insertDebugMarker(String&& markerLabel)
{
    auto sendResult = send(Messages::RemoteCommandEncoder::InsertDebugMarker(WTFMove(markerLabel)));
    UNUSED_VARIABLE(sendResult);
}

void RemoteCommandEncoderProxy::writeTimestamp(const WebCore::WebGPU::QuerySet& querySet, WebCore::WebGPU::Size32 queryIndex)
{
    auto convertedQuerySet = protectedConvertToBackingContext()->convertToBacking(querySet);

    auto sendResult = send(Messages::RemoteCommandEncoder::WriteTimestamp(convertedQuerySet, queryIndex));
    UNUSED_VARIABLE(sendResult);
}

void RemoteCommandEncoderProxy::resolveQuerySet(
    const WebCore::WebGPU::QuerySet& querySet,
    WebCore::WebGPU::Size32 firstQuery,
    WebCore::WebGPU::Size32 queryCount,
    const WebCore::WebGPU::Buffer& destination,
    WebCore::WebGPU::Size64 destinationOffset)
{
    Ref convertToBackingContext = m_convertToBackingContext;
    auto convertedQuerySet = convertToBackingContext->convertToBacking(querySet);
    auto convertedDestination = convertToBackingContext->convertToBacking(destination);

    auto sendResult = send(Messages::RemoteCommandEncoder::ResolveQuerySet(convertedQuerySet, firstQuery, queryCount, convertedDestination, destinationOffset));
    UNUSED_VARIABLE(sendResult);
}

RefPtr<WebCore::WebGPU::CommandBuffer> RemoteCommandEncoderProxy::finish(const WebCore::WebGPU::CommandBufferDescriptor& descriptor)
{
    Ref convertToBackingContext = m_convertToBackingContext;
    auto convertedDescriptor = convertToBackingContext->convertToBacking(descriptor);

    if (!convertedDescriptor)
        return nullptr;

    auto identifier = WebGPUIdentifier::generate();
    auto sendResult = send(Messages::RemoteCommandEncoder::Finish(*convertedDescriptor, identifier));
    if (sendResult != IPC::Error::NoError)
        return nullptr;

    auto result = RemoteCommandBufferProxy::create(m_root, convertToBackingContext, identifier);
    result->setLabel(WTFMove(convertedDescriptor->label));
    return result;
}

void RemoteCommandEncoderProxy::setLabelInternal(const String& label)
{
    auto sendResult = send(Messages::RemoteCommandEncoder::SetLabel(label));
    UNUSED_VARIABLE(sendResult);
}

Ref<ConvertToBackingContext> RemoteCommandEncoderProxy::protectedConvertToBackingContext() const
{
    return m_convertToBackingContext;
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
