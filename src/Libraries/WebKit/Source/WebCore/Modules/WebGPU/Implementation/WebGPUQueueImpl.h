/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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

#include "WebGPUPtr.h"
#include "WebGPUQueue.h"
#include <WebGPU/WebGPU.h>
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class QueueImpl final : public Queue {
    WTF_MAKE_TZONE_ALLOCATED(QueueImpl);
public:
    static Ref<QueueImpl> create(WebGPUPtr<WGPUQueue>&& queue, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new QueueImpl(WTFMove(queue), convertToBackingContext));
    }

    virtual ~QueueImpl();

private:
    friend class DowncastConvertToBackingContext;

    QueueImpl(WebGPUPtr<WGPUQueue>&&, ConvertToBackingContext&);

    QueueImpl(const QueueImpl&) = delete;
    QueueImpl(QueueImpl&&) = delete;
    QueueImpl& operator=(const QueueImpl&) = delete;
    QueueImpl& operator=(QueueImpl&&) = delete;

    WGPUQueue backing() const { return m_backing.get(); }

    void submit(Vector<Ref<WebGPU::CommandBuffer>>&&) final;

    void onSubmittedWorkDone(CompletionHandler<void()>&&) final;

    void writeBuffer(
        const Buffer&,
        Size64 bufferOffset,
        std::span<const uint8_t> source,
        Size64 dataOffset,
        std::optional<Size64>) final;

    void writeTexture(
        const ImageCopyTexture& destination,
        std::span<const uint8_t> source,
        const ImageDataLayout&,
        const Extent3D& size) final;

    void writeBufferNoCopy(
        const Buffer&,
        Size64 bufferOffset,
        std::span<uint8_t> source,
        Size64 dataOffset,
        std::optional<Size64>) final;

    void writeTexture(
        const ImageCopyTexture& destination,
        std::span<uint8_t> source,
        const ImageDataLayout&,
        const Extent3D& size) final;

    void copyExternalImageToTexture(
        const ImageCopyExternalImage& source,
        const ImageCopyTextureTagged& destination,
        const Extent3D& copySize) final;

    void setLabelInternal(const String&) final;
    RefPtr<WebCore::NativeImage> getNativeImage(WebCore::VideoFrame&) final;

    WebGPUPtr<WGPUQueue> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
