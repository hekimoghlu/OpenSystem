/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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

#if ENABLE(GPU_PROCESS)

#include "RemoteCommandEncoderProxy.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPURenderPassEncoder.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteRenderPassEncoderProxy final : public WebCore::WebGPU::RenderPassEncoder {
    WTF_MAKE_TZONE_ALLOCATED(RemoteRenderPassEncoderProxy);
public:
    static Ref<RemoteRenderPassEncoderProxy> create(RemoteCommandEncoderProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteRenderPassEncoderProxy(parent, convertToBackingContext, identifier));
    }

    virtual ~RemoteRenderPassEncoderProxy();

    RemoteGPUProxy& root() { return m_root; }

private:
    friend class DowncastConvertToBackingContext;

    RemoteRenderPassEncoderProxy(RemoteCommandEncoderProxy&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteRenderPassEncoderProxy(const RemoteRenderPassEncoderProxy&) = delete;
    RemoteRenderPassEncoderProxy(RemoteRenderPassEncoderProxy&&) = delete;
    RemoteRenderPassEncoderProxy& operator=(const RemoteRenderPassEncoderProxy&) = delete;
    RemoteRenderPassEncoderProxy& operator=(RemoteRenderPassEncoderProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }
    Ref<ConvertToBackingContext> protectedConvertToBackingContext() const;
    
    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(WTFMove(message), backing());
    }

    void setPipeline(const WebCore::WebGPU::RenderPipeline&) final;

    void setIndexBuffer(const WebCore::WebGPU::Buffer&, WebCore::WebGPU::IndexFormat, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64>) final;
    void setVertexBuffer(WebCore::WebGPU::Index32 slot, const WebCore::WebGPU::Buffer*, std::optional<WebCore::WebGPU::Size64> offset, std::optional<WebCore::WebGPU::Size64>) final;

    void draw(WebCore::WebGPU::Size32 vertexCount, std::optional<WebCore::WebGPU::Size32> instanceCount,
        std::optional<WebCore::WebGPU::Size32> firstVertex, std::optional<WebCore::WebGPU::Size32> firstInstance) final;
    void drawIndexed(WebCore::WebGPU::Size32 indexCount, std::optional<WebCore::WebGPU::Size32> instanceCount,
        std::optional<WebCore::WebGPU::Size32> firstIndex,
        std::optional<WebCore::WebGPU::SignedOffset32> baseVertex,
        std::optional<WebCore::WebGPU::Size32> firstInstance) final;

    void drawIndirect(const WebCore::WebGPU::Buffer& indirectBuffer, WebCore::WebGPU::Size64 indirectOffset) final;
    void drawIndexedIndirect(const WebCore::WebGPU::Buffer& indirectBuffer, WebCore::WebGPU::Size64 indirectOffset) final;

    void setBindGroup(WebCore::WebGPU::Index32, const WebCore::WebGPU::BindGroup&,
        std::optional<Vector<WebCore::WebGPU::BufferDynamicOffset>>&& dynamicOffsets) final;

    void setBindGroup(WebCore::WebGPU::Index32, const WebCore::WebGPU::BindGroup&,
        std::span<const uint32_t> dynamicOffsetsArrayBuffer,
        WebCore::WebGPU::Size64 dynamicOffsetsDataStart,
        WebCore::WebGPU::Size32 dynamicOffsetsDataLength) final;

    void pushDebugGroup(String&& groupLabel) final;
    void popDebugGroup() final;
    void insertDebugMarker(String&& markerLabel) final;

    void setViewport(float x, float y,
        float width, float height,
        float minDepth, float maxDepth) final;

    void setScissorRect(WebCore::WebGPU::IntegerCoordinate x, WebCore::WebGPU::IntegerCoordinate y,
        WebCore::WebGPU::IntegerCoordinate width, WebCore::WebGPU::IntegerCoordinate height) final;

    void setBlendConstant(WebCore::WebGPU::Color) final;
    void setStencilReference(WebCore::WebGPU::StencilValue) final;

    void beginOcclusionQuery(WebCore::WebGPU::Size32 queryIndex) final;
    void endOcclusionQuery() final;

    void executeBundles(Vector<Ref<WebCore::WebGPU::RenderBundle>>&&) final;
    void end() final;

    void setLabelInternal(const String&) final;

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteGPUProxy> m_root;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
