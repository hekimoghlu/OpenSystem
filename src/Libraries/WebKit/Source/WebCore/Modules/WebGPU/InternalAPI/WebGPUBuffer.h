/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
#include "WebGPUMapMode.h"
#include <cstdint>
#include <optional>
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore::WebGPU {

class Buffer : public RefCountedAndCanMakeWeakPtr<Buffer> {
public:
    virtual ~Buffer() = default;

    String label() const { return m_label; }

    void setLabel(String&& label)
    {
        m_label = WTFMove(label);
        setLabelInternal(m_label);
    }

    virtual void mapAsync(MapModeFlags, Size64 offset, std::optional<Size64>, CompletionHandler<void(bool)>&&) = 0;
    virtual void getMappedRange(Size64 offset, std::optional<Size64>, Function<void(std::span<uint8_t>)>&&) = 0;
    virtual void unmap() = 0;

    virtual void destroy() = 0;
    virtual std::span<uint8_t> getBufferContents() = 0;
    virtual void copyFrom(std::span<const uint8_t>, size_t offset) = 0;
protected:
    Buffer() = default;

private:
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer& operator=(Buffer&&) = delete;

    virtual void setLabelInternal(const String&) = 0;

    String m_label;
};

} // namespace WebCore::WebGPU
