/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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

#include "WebGPUColor.h"
#include "WebGPUIndexFormat.h"
#include "WebGPUIntegralTypes.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {

class BindGroup;
class Buffer;
class QuerySet;
class RenderBundle;
class RenderPipeline;

class RenderPassEncoder : public RefCountedAndCanMakeWeakPtr<RenderPassEncoder> {
public:
    virtual ~RenderPassEncoder() = default;

    String label() const { return m_label; }

    void setLabel(String&& label)
    {
        m_label = WTFMove(label);
        setLabelInternal(m_label);
    }

    virtual void setPipeline(const RenderPipeline&) = 0;

    virtual void setIndexBuffer(const Buffer&, IndexFormat, std::optional<Size64> offset, std::optional<Size64>) = 0;
    virtual void setVertexBuffer(Index32 slot, const Buffer*, std::optional<Size64> offset, std::optional<Size64>) = 0;

    virtual void draw(Size32 vertexCount, std::optional<Size32> instanceCount,
        std::optional<Size32> firstVertex, std::optional<Size32> firstInstance) = 0;
    virtual void drawIndexed(Size32 indexCount, std::optional<Size32> instanceCount,
        std::optional<Size32> firstIndex,
        std::optional<SignedOffset32> baseVertex,
        std::optional<Size32> firstInstance) = 0;

    virtual void drawIndirect(const Buffer& indirectBuffer, Size64 indirectOffset) = 0;
    virtual void drawIndexedIndirect(const Buffer& indirectBuffer, Size64 indirectOffset) = 0;

    virtual void setBindGroup(Index32, const BindGroup&,
        std::optional<Vector<BufferDynamicOffset>>&& dynamicOffsets) = 0;

    virtual void setBindGroup(Index32, const BindGroup&,
        std::span<const uint32_t> dynamicOffsetsArrayBuffer,
        Size64 dynamicOffsetsDataStart,
        Size32 dynamicOffsetsDataLength) = 0;

    virtual void pushDebugGroup(String&& groupLabel) = 0;
    virtual void popDebugGroup() = 0;
    virtual void insertDebugMarker(String&& markerLabel) = 0;

    virtual void setViewport(float x, float y,
        float width, float height,
        float minDepth, float maxDepth) = 0;

    virtual void setScissorRect(IntegerCoordinate x, IntegerCoordinate y,
        IntegerCoordinate width, IntegerCoordinate height) = 0;

    virtual void setBlendConstant(Color) = 0;
    virtual void setStencilReference(StencilValue) = 0;

    virtual void beginOcclusionQuery(Size32 queryIndex) = 0;
    virtual void endOcclusionQuery() = 0;

    virtual void executeBundles(Vector<Ref<RenderBundle>>&&) = 0;
    virtual void end() = 0;

protected:
    RenderPassEncoder() = default;

private:
    RenderPassEncoder(const RenderPassEncoder&) = delete;
    RenderPassEncoder(RenderPassEncoder&&) = delete;
    RenderPassEncoder& operator=(const RenderPassEncoder&) = delete;
    RenderPassEncoder& operator=(RenderPassEncoder&&) = delete;

    virtual void setLabelInternal(const String&) = 0;

    String m_label;
};

} // namespace WebCore::WebGPU
