/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
#include <WebCore/WebGPUComputePassEncoder.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteComputePassEncoderProxy final : public WebCore::WebGPU::ComputePassEncoder {
    WTF_MAKE_TZONE_ALLOCATED(RemoteComputePassEncoderProxy);
public:
    static Ref<RemoteComputePassEncoderProxy> create(RemoteCommandEncoderProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteComputePassEncoderProxy(parent, convertToBackingContext, identifier));
    }

    virtual ~RemoteComputePassEncoderProxy();

    RemoteGPUProxy& root() { return m_root; }

private:
    friend class DowncastConvertToBackingContext;

    RemoteComputePassEncoderProxy(RemoteCommandEncoderProxy&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteComputePassEncoderProxy(const RemoteComputePassEncoderProxy&) = delete;
    RemoteComputePassEncoderProxy(RemoteComputePassEncoderProxy&&) = delete;
    RemoteComputePassEncoderProxy& operator=(const RemoteComputePassEncoderProxy&) = delete;
    RemoteComputePassEncoderProxy& operator=(RemoteComputePassEncoderProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }
    
    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(WTFMove(message), backing());
    }

    void setPipeline(const WebCore::WebGPU::ComputePipeline&) final;
    void dispatch(WebCore::WebGPU::Size32 workgroupCountX, WebCore::WebGPU::Size32 workgroupCountY = 1, WebCore::WebGPU::Size32 workgroupCountZ = 1) final;
    void dispatchIndirect(const WebCore::WebGPU::Buffer& indirectBuffer, WebCore::WebGPU::Size64 indirectOffset) final;

    void end() final;

    void setBindGroup(WebCore::WebGPU::Index32, const WebCore::WebGPU::BindGroup&,
        std::optional<Vector<WebCore::WebGPU::BufferDynamicOffset>>&&) final;

    void setBindGroup(WebCore::WebGPU::Index32, const WebCore::WebGPU::BindGroup&,
        std::span<const uint32_t> dynamicOffsetsArrayBuffer,
        WebCore::WebGPU::Size64 dynamicOffsetsDataStart,
        WebCore::WebGPU::Size32 dynamicOffsetsDataLength) final;

    void pushDebugGroup(String&& groupLabel) final;
    void popDebugGroup() final;
    void insertDebugMarker(String&& markerLabel) final;

    void setLabelInternal(const String&) final;

    Ref<ConvertToBackingContext> protectedConvertToBackingContext() const { return m_convertToBackingContext; }

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteGPUProxy> m_root;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
