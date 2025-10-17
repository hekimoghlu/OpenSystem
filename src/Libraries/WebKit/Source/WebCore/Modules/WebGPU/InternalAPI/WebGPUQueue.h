/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
#include "WebGPUExtent3D.h"
#include "WebGPUImageCopyExternalImage.h"
#include "WebGPUImageCopyTexture.h"
#include "WebGPUImageCopyTextureTagged.h"
#include "WebGPUImageDataLayout.h"
#include "WebGPUIntegralTypes.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class NativeImage;
class VideoFrame;
}

namespace WebCore::WebGPU {

class Buffer;

class Queue : public RefCountedAndCanMakeWeakPtr<Queue> {
public:
    virtual ~Queue() = default;

    String label() const { return m_label; }

    void setLabel(String&& label)
    {
        m_label = WTFMove(label);
        setLabelInternal(m_label);
    }

    virtual void submit(Vector<Ref<WebGPU::CommandBuffer>>&&) = 0;

    virtual void onSubmittedWorkDone(CompletionHandler<void()>&&) = 0;

    virtual void writeBuffer(
        const Buffer&,
        Size64 bufferOffset,
        std::span<const uint8_t> source,
        Size64 dataOffset = 0,
        std::optional<Size64> = std::nullopt) = 0;

    virtual void writeTexture(
        const ImageCopyTexture& destination,
        std::span<const uint8_t> source,
        const ImageDataLayout&,
        const Extent3D& size) = 0;

    virtual void writeBufferNoCopy(
        const Buffer&,
        Size64 bufferOffset,
        std::span<uint8_t> source,
        Size64 dataOffset = 0,
        std::optional<Size64> = std::nullopt) = 0;

    virtual void writeTexture(
        const ImageCopyTexture& destination,
        std::span<uint8_t> source,
        const ImageDataLayout&,
        const Extent3D& size) = 0;

    virtual void copyExternalImageToTexture(
        const ImageCopyExternalImage& source,
        const ImageCopyTextureTagged& destination,
        const Extent3D& copySize) = 0;

    virtual RefPtr<WebCore::NativeImage> getNativeImage(WebCore::VideoFrame&) = 0;
protected:
    Queue() = default;

private:
    Queue(const Queue&) = delete;
    Queue(Queue&&) = delete;
    Queue& operator=(const Queue&) = delete;
    Queue& operator=(Queue&&) = delete;

    virtual void setLabelInternal(const String&) = 0;

    String m_label;
};

} // namespace WebCore::WebGPU
