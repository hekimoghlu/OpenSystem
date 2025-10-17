/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
#pragma once

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUCommandEncoder.h"
#include "WebGPUPtr.h"
#include <WebGPU/WebGPU.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class CommandEncoderImpl final : public CommandEncoder {
    WTF_MAKE_TZONE_ALLOCATED(CommandEncoderImpl);
public:
    static Ref<CommandEncoderImpl> create(WebGPUPtr<WGPUCommandEncoder>&& commandEncoder, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new CommandEncoderImpl(WTFMove(commandEncoder), convertToBackingContext));
    }

    virtual ~CommandEncoderImpl();

private:
    friend class DowncastConvertToBackingContext;

    CommandEncoderImpl(WebGPUPtr<WGPUCommandEncoder>&&, ConvertToBackingContext&);

    CommandEncoderImpl(const CommandEncoderImpl&) = delete;
    CommandEncoderImpl(CommandEncoderImpl&&) = delete;
    CommandEncoderImpl& operator=(const CommandEncoderImpl&) = delete;
    CommandEncoderImpl& operator=(CommandEncoderImpl&&) = delete;

    WGPUCommandEncoder backing() const { return m_backing.get(); }

    RefPtr<RenderPassEncoder> beginRenderPass(const RenderPassDescriptor&) final;
    RefPtr<ComputePassEncoder> beginComputePass(const std::optional<ComputePassDescriptor>&) final;

    Ref<ConvertToBackingContext> protectedConvertToBackingContext() const { return m_convertToBackingContext; }

    void copyBufferToBuffer(
        const Buffer& source,
        Size64 sourceOffset,
        const Buffer& destination,
        Size64 destinationOffset,
        Size64) final;

    void copyBufferToTexture(
        const ImageCopyBuffer& source,
        const ImageCopyTexture& destination,
        const Extent3D& copySize) final;

    void copyTextureToBuffer(
        const ImageCopyTexture& source,
        const ImageCopyBuffer& destination,
        const Extent3D& copySize) final;

    void copyTextureToTexture(
        const ImageCopyTexture& source,
        const ImageCopyTexture& destination,
        const Extent3D& copySize) final;

    void clearBuffer(
        const Buffer&,
        Size64 offset,
        std::optional<Size64>) final;

    void pushDebugGroup(String&& groupLabel) final;
    void popDebugGroup() final;
    void insertDebugMarker(String&& markerLabel) final;

    void writeTimestamp(const QuerySet&, Size32 queryIndex) final;

    void resolveQuerySet(
        const QuerySet&,
        Size32 firstQuery,
        Size32 queryCount,
        const Buffer& destination,
        Size64 destinationOffset) final;

    RefPtr<CommandBuffer> finish(const CommandBufferDescriptor&) final;

    void setLabelInternal(const String&) final;

    WebGPUPtr<WGPUCommandEncoder> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
