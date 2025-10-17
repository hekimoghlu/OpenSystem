/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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

#include "RemoteDeviceProxy.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPUCommandEncoder.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteCommandEncoderProxy final : public WebCore::WebGPU::CommandEncoder {
    WTF_MAKE_TZONE_ALLOCATED(RemoteCommandEncoderProxy);
public:
    static Ref<RemoteCommandEncoderProxy> create(RemoteGPUProxy& root, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteCommandEncoderProxy(root, convertToBackingContext, identifier));
    }

    virtual ~RemoteCommandEncoderProxy();

    RemoteGPUProxy& root() { return m_root; }

private:
    friend class DowncastConvertToBackingContext;

    RemoteCommandEncoderProxy(RemoteGPUProxy&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteCommandEncoderProxy(const RemoteCommandEncoderProxy&) = delete;
    RemoteCommandEncoderProxy(RemoteCommandEncoderProxy&&) = delete;
    RemoteCommandEncoderProxy& operator=(const RemoteCommandEncoderProxy&) = delete;
    RemoteCommandEncoderProxy& operator=(RemoteCommandEncoderProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }
    
    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(WTFMove(message), backing());
    }
    template<typename T>
    WARN_UNUSED_RETURN IPC::Connection::SendSyncResult<T> sendSync(T&& message)
    {
        return root().protectedStreamClientConnection()->sendSync(WTFMove(message), backing());
    }

    RefPtr<WebCore::WebGPU::RenderPassEncoder> beginRenderPass(const WebCore::WebGPU::RenderPassDescriptor&) final;
    RefPtr<WebCore::WebGPU::ComputePassEncoder> beginComputePass(const std::optional<WebCore::WebGPU::ComputePassDescriptor>&) final;

    void copyBufferToBuffer(
        const WebCore::WebGPU::Buffer& source,
        WebCore::WebGPU::Size64 sourceOffset,
        const WebCore::WebGPU::Buffer& destination,
        WebCore::WebGPU::Size64 destinationOffset,
        WebCore::WebGPU::Size64) final;

    void copyBufferToTexture(
        const WebCore::WebGPU::ImageCopyBuffer& source,
        const WebCore::WebGPU::ImageCopyTexture& destination,
        const WebCore::WebGPU::Extent3D& copySize) final;

    void copyTextureToBuffer(
        const WebCore::WebGPU::ImageCopyTexture& source,
        const WebCore::WebGPU::ImageCopyBuffer& destination,
        const WebCore::WebGPU::Extent3D& copySize) final;

    void copyTextureToTexture(
        const WebCore::WebGPU::ImageCopyTexture& source,
        const WebCore::WebGPU::ImageCopyTexture& destination,
        const WebCore::WebGPU::Extent3D& copySize) final;

    void clearBuffer(
        const WebCore::WebGPU::Buffer&,
        WebCore::WebGPU::Size64 offset = 0,
        std::optional<WebCore::WebGPU::Size64> = std::nullopt) final;

    void pushDebugGroup(String&& groupLabel) final;
    void popDebugGroup() final;
    void insertDebugMarker(String&& markerLabel) final;

    void writeTimestamp(const WebCore::WebGPU::QuerySet&, WebCore::WebGPU::Size32 queryIndex) final;

    void resolveQuerySet(
        const WebCore::WebGPU::QuerySet&,
        WebCore::WebGPU::Size32 firstQuery,
        WebCore::WebGPU::Size32 queryCount,
        const WebCore::WebGPU::Buffer& destination,
        WebCore::WebGPU::Size64 destinationOffset) final;

    RefPtr<WebCore::WebGPU::CommandBuffer> finish(const WebCore::WebGPU::CommandBufferDescriptor&) final;

    void setLabelInternal(const String&) final;

    Ref<ConvertToBackingContext> protectedConvertToBackingContext() const;

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteGPUProxy> m_root;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
