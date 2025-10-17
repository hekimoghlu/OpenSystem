/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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

#include "WebGPUCommandBuffer.h"
#include "WebGPUCommandBufferDescriptor.h"
#include "WebGPUComputePassDescriptor.h"
#include "WebGPUComputePassEncoder.h"
#include "WebGPUExtent3D.h"
#include "WebGPUImageCopyBuffer.h"
#include "WebGPUImageCopyTexture.h"
#include "WebGPUIntegralTypes.h"
#include "WebGPURenderPassDescriptor.h"
#include "WebGPURenderPassEncoder.h"
#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {

class Buffer;
class QuerySet;

class CommandEncoder : public RefCountedAndCanMakeWeakPtr<CommandEncoder> {
public:
    virtual ~CommandEncoder() = default;

    String label() const { return m_label; }

    void setLabel(String&& label)
    {
        m_label = WTFMove(label);
        setLabelInternal(m_label);
    }

    virtual RefPtr<RenderPassEncoder> beginRenderPass(const RenderPassDescriptor&) = 0;
    virtual RefPtr<ComputePassEncoder> beginComputePass(const std::optional<ComputePassDescriptor>&) = 0;

    virtual void copyBufferToBuffer(
        const Buffer& source,
        Size64 sourceOffset,
        const Buffer& destination,
        Size64 destinationOffset,
        Size64) = 0;

    virtual void copyBufferToTexture(
        const ImageCopyBuffer& source,
        const ImageCopyTexture& destination,
        const Extent3D& copySize) = 0;

    virtual void copyTextureToBuffer(
        const ImageCopyTexture& source,
        const ImageCopyBuffer& destination,
        const Extent3D& copySize) = 0;

    virtual void copyTextureToTexture(
        const ImageCopyTexture& source,
        const ImageCopyTexture& destination,
        const Extent3D& copySize) = 0;

    virtual void clearBuffer(
        const Buffer&,
        Size64 offset = 0,
        std::optional<Size64> = std::nullopt) = 0;

    virtual void pushDebugGroup(String&& groupLabel) = 0;
    virtual void popDebugGroup() = 0;
    virtual void insertDebugMarker(String&& markerLabel) = 0;

    virtual void writeTimestamp(const QuerySet&, Size32 queryIndex) = 0;

    virtual void resolveQuerySet(
        const QuerySet&,
        Size32 firstQuery,
        Size32 queryCount,
        const Buffer& destination,
        Size64 destinationOffset) = 0;

    virtual RefPtr<CommandBuffer> finish(const CommandBufferDescriptor&) = 0;

protected:
    CommandEncoder() = default;

private:
    CommandEncoder(const CommandEncoder&) = delete;
    CommandEncoder(CommandEncoder&&) = delete;
    CommandEncoder& operator=(const CommandEncoder&) = delete;
    CommandEncoder& operator=(CommandEncoder&&) = delete;

    virtual void setLabelInternal(const String&) = 0;

    String m_label;
};

} // namespace WebCore::WebGPU
