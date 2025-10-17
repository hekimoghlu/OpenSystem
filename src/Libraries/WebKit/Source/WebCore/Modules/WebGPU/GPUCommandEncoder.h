/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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

#include "GPUCommandBufferDescriptor.h"
#include "GPUComputePassDescriptor.h"
#include "GPUComputePassEncoder.h"
#include "GPUExtent3DDict.h"
#include "GPUImageCopyBuffer.h"
#include "GPUImageCopyTexture.h"
#include "GPUIntegralTypes.h"
#include "GPURenderPassDescriptor.h"
#include "GPURenderPassEncoder.h"
#include "WebGPUCommandEncoder.h"
#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUBuffer;
class GPUCommandBuffer;
class GPUQuerySet;

namespace WebGPU {
class Device;
}

class GPUCommandEncoder : public RefCounted<GPUCommandEncoder> {
public:
    static Ref<GPUCommandEncoder> create(Ref<WebGPU::CommandEncoder>&& backing, WebGPU::Device& device)
    {
        return adoptRef(*new GPUCommandEncoder(WTFMove(backing), device));
    }

    String label() const;
    void setLabel(String&&);

    ExceptionOr<Ref<GPURenderPassEncoder>> beginRenderPass(const GPURenderPassDescriptor&);
    ExceptionOr<Ref<GPUComputePassEncoder>> beginComputePass(const std::optional<GPUComputePassDescriptor>&);

    void copyBufferToBuffer(
        const GPUBuffer& source,
        GPUSize64 sourceOffset,
        const GPUBuffer& destination,
        GPUSize64 destinationOffset,
        GPUSize64);

    void copyBufferToTexture(
        const GPUImageCopyBuffer& source,
        const GPUImageCopyTexture& destination,
        const GPUExtent3D& copySize);

    void copyTextureToBuffer(
        const GPUImageCopyTexture& source,
        const GPUImageCopyBuffer& destination,
        const GPUExtent3D& copySize);

    void copyTextureToTexture(
        const GPUImageCopyTexture& source,
        const GPUImageCopyTexture& destination,
        const GPUExtent3D& copySize);

    void clearBuffer(
        const GPUBuffer&,
        std::optional<GPUSize64> offset,
        std::optional<GPUSize64>);

    void pushDebugGroup(String&& groupLabel);
    void popDebugGroup();
    void insertDebugMarker(String&& markerLabel);

    void writeTimestamp(const GPUQuerySet&, GPUSize32 queryIndex);

    void resolveQuerySet(
        const GPUQuerySet&,
        GPUSize32 firstQuery,
        GPUSize32 queryCount,
        const GPUBuffer& destination,
        GPUSize64 destinationOffset);

    ExceptionOr<Ref<GPUCommandBuffer>> finish(const std::optional<GPUCommandBufferDescriptor>&);

    WebGPU::CommandEncoder& backing() { return m_backing; }
    const WebGPU::CommandEncoder& backing() const { return m_backing; }
    void setBacking(WebGPU::CommandEncoder&);

private:
    GPUCommandEncoder(Ref<WebGPU::CommandEncoder>&&, WebGPU::Device&);

    Ref<WebGPU::CommandEncoder> m_backing;
    WeakPtr<WebGPU::Device> m_device;
};

}
