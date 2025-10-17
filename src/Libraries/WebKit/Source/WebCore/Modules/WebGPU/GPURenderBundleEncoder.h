/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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
#include "GPUIndexFormat.h"
#include "GPUIntegralTypes.h"
#include "GPURenderBundleDescriptor.h"
#include "WebGPURenderBundleEncoder.h"
#include <JavaScriptCore/Uint32Array.h>
#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUBindGroup;
class GPUBuffer;
class GPURenderBundle;
class GPURenderPipeline;

class GPURenderBundleEncoder : public RefCounted<GPURenderBundleEncoder> {
public:
    static Ref<GPURenderBundleEncoder> create(Ref<WebGPU::RenderBundleEncoder>&& backing)
    {
        return adoptRef(*new GPURenderBundleEncoder(WTFMove(backing)));
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

    ExceptionOr<Ref<GPURenderBundle>> finish(const std::optional<GPURenderBundleDescriptor>&);

    WebGPU::RenderBundleEncoder& backing() { return m_backing; }
    const WebGPU::RenderBundleEncoder& backing() const { return m_backing; }

private:
    GPURenderBundleEncoder(Ref<WebGPU::RenderBundleEncoder>&& backing)
        : m_backing(WTFMove(backing))
    {
    }

    Ref<WebGPU::RenderBundleEncoder> m_backing;
};

}
