/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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

#include "WebGPUIntegralTypes.h"
#include <cstdint>
#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {

class BindGroup;
class Buffer;
class ComputePipeline;
class QuerySet;

class ComputePassEncoder : public RefCountedAndCanMakeWeakPtr<ComputePassEncoder> {
public:
    virtual ~ComputePassEncoder() = default;

    String label() const { return m_label; }

    void setLabel(String&& label)
    {
        m_label = WTFMove(label);
        setLabelInternal(m_label);
    }

    virtual void setPipeline(const ComputePipeline&) = 0;
    virtual void dispatch(Size32 workgroupCountX, Size32 workgroupCountY = 1, Size32 workgroupCountZ = 1) = 0;
    virtual void dispatchIndirect(const Buffer& indirectBuffer, Size64 indirectOffset) = 0;

    virtual void end() = 0;

    virtual void setBindGroup(Index32, const BindGroup&,
        std::optional<Vector<BufferDynamicOffset>>&&) = 0;

    virtual void setBindGroup(Index32, const BindGroup&,
        std::span<const uint32_t> dynamicOffsetsArrayBuffer,
        Size64 dynamicOffsetsDataStart,
        Size32 dynamicOffsetsDataLength) = 0;

    virtual void pushDebugGroup(String&& groupLabel) = 0;
    virtual void popDebugGroup() = 0;
    virtual void insertDebugMarker(String&& markerLabel) = 0;

protected:
    ComputePassEncoder() = default;

private:
    ComputePassEncoder(const ComputePassEncoder&) = delete;
    ComputePassEncoder(ComputePassEncoder&&) = delete;
    ComputePassEncoder& operator=(const ComputePassEncoder&) = delete;
    ComputePassEncoder& operator=(ComputePassEncoder&&) = delete;

    virtual void setLabelInternal(const String&) = 0;

    String m_label;
};

} // namespace WebCore::WebGPU
