/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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

#include "WebGPUBuffer.h"
#include "WebGPUPtr.h"
#include <WebGPU/WebGPU.h>
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class BufferImpl final : public Buffer {
    WTF_MAKE_TZONE_ALLOCATED(BufferImpl);
public:
    static Ref<BufferImpl> create(WebGPUPtr<WGPUBuffer>&& buffer, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new BufferImpl(WTFMove(buffer), convertToBackingContext));
    }

    virtual ~BufferImpl();

private:
    friend class DowncastConvertToBackingContext;

    BufferImpl(WebGPUPtr<WGPUBuffer>&&, ConvertToBackingContext&);

    BufferImpl(const BufferImpl&) = delete;
    BufferImpl(BufferImpl&&) = delete;
    BufferImpl& operator=(const BufferImpl&) = delete;
    BufferImpl& operator=(BufferImpl&&) = delete;

    WGPUBuffer backing() const { return m_backing.get(); }

    void mapAsync(MapModeFlags, Size64 offset, std::optional<Size64> sizeForMap, CompletionHandler<void(bool)>&&) final;
    void getMappedRange(Size64 offset, std::optional<Size64>, Function<void(std::span<uint8_t>)>&&) final;
    std::span<uint8_t> getBufferContents() final;
    void unmap() final;
    void copyFrom(std::span<const uint8_t>, size_t offset) final;

    void destroy() final;

    void setLabelInternal(const String&) final;

    WebGPUPtr<WGPUBuffer> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
