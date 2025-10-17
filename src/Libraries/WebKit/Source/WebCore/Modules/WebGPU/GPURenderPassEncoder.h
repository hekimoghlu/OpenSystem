/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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

#include "ExceptionOr.h"
#include "GPUColorDict.h"
#include "GPUIndexFormat.h"
#include "GPUIntegralTypes.h"
#include "WebGPURenderPassEncoder.h"
#include <JavaScriptCore/Uint32Array.h>
#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUBindGroup;
class GPUBuffer;
class GPUQuerySet;
class GPURenderBundle;
class GPURenderPipeline;

namespace WebGPU {
class Device;
}

class GPURenderPassEncoder : public RefCounted<GPURenderPassEncoder> {
public:
    static Ref<GPURenderPassEncoder> create(Ref<WebGPU::RenderPassEncoder>&& backing, WebGPU::Device& device)
    {
        return adoptRef(*new GPURenderPassEncoder(WTFMove(backing), device));
    }

    String label() const;
    void setLabel(String&&);

    void setPipeline(const GPURenderPipeline&);

    void setIndexBuffer(const GPUBuffer&, GPUIndexFormat, std::optional<GPUSize64> offset, std::optional<GPUSize64>);
    void setVertexBuffer(GPUIndex32 slot, const GPUBuffer*, std::optional<GPUSize64> offset, std::optional<GPUSize64>);

    void draw(GPUSize32 vertexCount, std::optional<GPUSize32> instanceCount,
        std::optional<GPUSize32> firstVertex, std::optional<GPUSize32> firstInstance);
    void drawIndexed(GPUSize32 indexCount, std::optional<GPUSize32> instanceCount,
        std::optional<GPUSize32> firstIndex,
        std::optional<GPUSignedOffset32> baseVertex,
        std::optional<GPUSize32> firstInstance);

    void drawIndirect(const GPUBuffer& indirectBuffer, GPUSize64 indirectOffset);
    void drawIndexedIndirect(const GPUBuffer& indirectBuffer, GPUSize64 indirectOffset);

    void setBindGroup(GPUIndex32, const GPUBindGroup&,
        std::optional<Vector<GPUBufferDynamicOffset>>&& dynamicOffsets);

    ExceptionOr<void> setBindGroup(GPUIndex32, const GPUBindGroup&,
        const Uint32Array& dynamicOffsetsData,
        GPUSize64 dynamicOffsetsDataStart,
        GPUSize32 dynamicOffsetsDataLength);

    void pushDebugGroup(String&& groupLabel);
    void popDebugGroup();
    void insertDebugMarker(String&& markerLabel);

    void setViewport(float x, float y,
        float width, float height,
        float minDepth, float maxDepth);

    void setScissorRect(GPUIntegerCoordinate x, GPUIntegerCoordinate y,
        GPUIntegerCoordinate width, GPUIntegerCoordinate height);

    void setBlendConstant(GPUColor);
    void setStencilReference(GPUStencilValue);

    void beginOcclusionQuery(GPUSize32 queryIndex);
    void endOcclusionQuery();

    void executeBundles(Vector<Ref<GPURenderBundle>>&&);
    void end();

    WebGPU::RenderPassEncoder& backing() { return m_backing; }
    const WebGPU::RenderPassEncoder& backing() const { return m_backing; }

private:
    GPURenderPassEncoder(Ref<WebGPU::RenderPassEncoder>&& backing, WebGPU::Device&);

    Ref<WebGPU::RenderPassEncoder> m_backing;
    WeakPtr<WebGPU::Device> m_device;
};

}
