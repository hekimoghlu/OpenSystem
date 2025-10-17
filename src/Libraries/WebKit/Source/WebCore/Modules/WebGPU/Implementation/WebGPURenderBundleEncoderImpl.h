/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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

#include "WebGPUPtr.h"
#include "WebGPURenderBundleEncoder.h"
#include <WebGPU/WebGPU.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class RenderBundleEncoderImpl final : public RenderBundleEncoder {
    WTF_MAKE_TZONE_ALLOCATED(RenderBundleEncoderImpl);
public:
    static Ref<RenderBundleEncoderImpl> create(WebGPUPtr<WGPURenderBundleEncoder>&& renderBundleEncoder, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new RenderBundleEncoderImpl(WTFMove(renderBundleEncoder), convertToBackingContext));
    }

    virtual ~RenderBundleEncoderImpl();

private:
    friend class DowncastConvertToBackingContext;

    RenderBundleEncoderImpl(WebGPUPtr<WGPURenderBundleEncoder>&&, ConvertToBackingContext&);

    RenderBundleEncoderImpl(const RenderBundleEncoderImpl&) = delete;
    RenderBundleEncoderImpl(RenderBundleEncoderImpl&&) = delete;
    RenderBundleEncoderImpl& operator=(const RenderBundleEncoderImpl&) = delete;
    RenderBundleEncoderImpl& operator=(RenderBundleEncoderImpl&&) = delete;

    WGPURenderBundleEncoder backing() const { return m_backing.get(); }

    void setPipeline(const RenderPipeline&) final;

    void setIndexBuffer(const Buffer&, IndexFormat, std::optional<Size64> offset, std::optional<Size64>) final;
    void setVertexBuffer(Index32 slot, const Buffer*, std::optional<Size64> offset, std::optional<Size64>) final;

    void draw(Size32 vertexCount, std::optional<Size32> instanceCount,
        std::optional<Size32> firstVertex, std::optional<Size32> firstInstance) final;
    void drawIndexed(Size32 indexCount, std::optional<Size32> instanceCount,
        std::optional<Size32> firstIndex,
        std::optional<SignedOffset32> baseVertex,
        std::optional<Size32> firstInstance) final;

    void drawIndirect(const Buffer& indirectBuffer, Size64 indirectOffset) final;
    void drawIndexedIndirect(const Buffer& indirectBuffer, Size64 indirectOffset) final;

    void setBindGroup(Index32, const BindGroup&,
        std::optional<Vector<BufferDynamicOffset>>&& dynamicOffsets) final;

    void setBindGroup(Index32, const BindGroup&,
        std::span<const uint32_t> dynamicOffsetsArrayBuffer,
        Size64 dynamicOffsetsDataStart,
        Size32 dynamicOffsetsDataLength) final;

    void pushDebugGroup(String&& groupLabel) final;
    void popDebugGroup() final;
    void insertDebugMarker(String&& markerLabel) final;

    RefPtr<RenderBundle> finish(const RenderBundleDescriptor&) final;

    void setLabelInternal(const String&) final;

    Ref<ConvertToBackingContext> protectedConvertToBackingContext() const { return m_convertToBackingContext; }

    WebGPUPtr<WGPURenderBundleEncoder> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
