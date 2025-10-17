/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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

#include "RemoteAdapterProxy.h"
#include "RemoteVideoFrameObjectHeapProxy.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPUQueue.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit::WebGPU {

class ConvertToBackingContext;

class RemoteQueueProxy final : public WebCore::WebGPU::Queue {
    WTF_MAKE_TZONE_ALLOCATED(RemoteQueueProxy);
public:
    static Ref<RemoteQueueProxy> create(RemoteAdapterProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier)
    {
        return adoptRef(*new RemoteQueueProxy(parent, convertToBackingContext, identifier));
    }

    virtual ~RemoteQueueProxy();

    RemoteAdapterProxy& parent() { return m_parent; }
    RemoteGPUProxy& root() { return m_parent->root(); }
    void submit(Vector<Ref<WebCore::WebGPU::CommandBuffer>>&&) final;

private:
    friend class DowncastConvertToBackingContext;

    RemoteQueueProxy(RemoteAdapterProxy&, ConvertToBackingContext&, WebGPUIdentifier);

    RemoteQueueProxy(const RemoteQueueProxy&) = delete;
    RemoteQueueProxy(RemoteQueueProxy&&) = delete;
    RemoteQueueProxy& operator=(const RemoteQueueProxy&) = delete;
    RemoteQueueProxy& operator=(RemoteQueueProxy&&) = delete;

    WebGPUIdentifier backing() const { return m_backing; }

    Ref<ConvertToBackingContext> protectedConvertToBackingContext() const;
    
    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(WTFMove(message), backing());
    }
    template<typename T, typename C>
    WARN_UNUSED_RETURN std::optional<IPC::StreamClientConnection::AsyncReplyID> sendWithAsyncReply(T&& message, C&& completionHandler)
    {
        return root().protectedStreamClientConnection()->sendWithAsyncReply(WTFMove(message), completionHandler, backing());
    }

    void onSubmittedWorkDone(CompletionHandler<void()>&&) final;

    void writeBuffer(
        const WebCore::WebGPU::Buffer&,
        WebCore::WebGPU::Size64 bufferOffset,
        std::span<const uint8_t> source,
        WebCore::WebGPU::Size64 dataOffset = 0,
        std::optional<WebCore::WebGPU::Size64> = std::nullopt) final;

    void writeTexture(
        const WebCore::WebGPU::ImageCopyTexture& destination,
        std::span<const uint8_t> source,
        const WebCore::WebGPU::ImageDataLayout&,
        const WebCore::WebGPU::Extent3D& size) final;

    void writeBufferNoCopy(
        const WebCore::WebGPU::Buffer&,
        WebCore::WebGPU::Size64 bufferOffset,
        std::span<uint8_t> source,
        WebCore::WebGPU::Size64 dataOffset = 0,
        std::optional<WebCore::WebGPU::Size64> = std::nullopt) final;

    void writeTexture(
        const WebCore::WebGPU::ImageCopyTexture& destination,
        std::span<uint8_t> source,
        const WebCore::WebGPU::ImageDataLayout&,
        const WebCore::WebGPU::Extent3D& size) final;

    void copyExternalImageToTexture(
        const WebCore::WebGPU::ImageCopyExternalImage& source,
        const WebCore::WebGPU::ImageCopyTextureTagged& destination,
        const WebCore::WebGPU::Extent3D& copySize) final;

    void setLabelInternal(const String&) final;
#if ENABLE(VIDEO)
    RefPtr<RemoteVideoFrameObjectHeapProxy> protectedVideoFrameObjectHeapProxy() const;
#endif
    RefPtr<WebCore::NativeImage> getNativeImage(WebCore::VideoFrame&) final;

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteAdapterProxy> m_parent;
#if ENABLE(VIDEO)
    RefPtr<RemoteVideoFrameObjectHeapProxy> m_videoFrameObjectHeapProxy;
#endif
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
