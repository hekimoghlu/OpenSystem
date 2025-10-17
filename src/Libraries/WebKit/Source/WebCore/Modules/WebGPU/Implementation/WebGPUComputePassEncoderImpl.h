/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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

#include "WebGPUComputePassEncoder.h"
#include "WebGPUPtr.h"
#include <WebGPU/WebGPU.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class ComputePassEncoderImpl final : public ComputePassEncoder {
    WTF_MAKE_TZONE_ALLOCATED(ComputePassEncoderImpl);
public:
    static Ref<ComputePassEncoderImpl> create(WebGPUPtr<WGPUComputePassEncoder>&& computePassEncoder, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new ComputePassEncoderImpl(WTFMove(computePassEncoder), convertToBackingContext));
    }

    virtual ~ComputePassEncoderImpl();

private:
    friend class DowncastConvertToBackingContext;

    ComputePassEncoderImpl(WebGPUPtr<WGPUComputePassEncoder>&&, ConvertToBackingContext&);

    ComputePassEncoderImpl(const ComputePassEncoderImpl&) = delete;
    ComputePassEncoderImpl(ComputePassEncoderImpl&&) = delete;
    ComputePassEncoderImpl& operator=(const ComputePassEncoderImpl&) = delete;
    ComputePassEncoderImpl& operator=(ComputePassEncoderImpl&&) = delete;

    WGPUComputePassEncoder backing() const { return m_backing.get(); }

    void setPipeline(const ComputePipeline&) final;
    void dispatch(Size32 workgroupCountX, Size32 workgroupCountY, Size32 workgroupCountZ) final;
    void dispatchIndirect(const Buffer& indirectBuffer, Size64 indirectOffset) final;

    void end() final;

    void setBindGroup(Index32, const BindGroup&,
        std::optional<Vector<BufferDynamicOffset>>&&) final;

    void setBindGroup(Index32, const BindGroup&,
        std::span<const uint32_t> dynamicOffsetsArrayBuffer,
        Size64 dynamicOffsetsDataStart,
        Size32 dynamicOffsetsDataLength) final;

    void pushDebugGroup(String&& groupLabel) final;
    void popDebugGroup() final;
    void insertDebugMarker(String&& markerLabel) final;

    void setLabelInternal(const String&) final;

    Ref<ConvertToBackingContext> protectedConvertToBackingContext() const { return m_convertToBackingContext; }

    WebGPUPtr<WGPUComputePassEncoder> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
