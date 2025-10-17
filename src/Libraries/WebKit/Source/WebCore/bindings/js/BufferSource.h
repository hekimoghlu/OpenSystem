/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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

#include <JavaScriptCore/ArrayBuffer.h>
#include <JavaScriptCore/ArrayBufferView.h>
#include <span>
#include <variant>
#include <wtf/Compiler.h>
#include <wtf/RefPtr.h>

#if PLATFORM(COCOA) && defined(__OBJC__)
#include <wtf/cocoa/SpanCocoa.h>
OBJC_CLASS NSData;
#endif

namespace WebCore {

class BufferSource {
public:
    using VariantType = std::variant<RefPtr<JSC::ArrayBufferView>, RefPtr<JSC::ArrayBuffer>>;

    BufferSource() { }
    BufferSource(VariantType&& variant)
        : m_variant(WTFMove(variant))
    { }

    BufferSource(const VariantType& variant)
        : m_variant(variant)
    { }
    explicit BufferSource(std::span<const uint8_t> span)
        : m_variant(JSC::ArrayBuffer::tryCreate(span)) { }

    const VariantType& variant() const { return m_variant; }

    size_t length() const
    {
        return std::visit([](auto& buffer) {
            return buffer ? buffer->byteLength() : 0;
        }, m_variant);
    }

    std::span<const uint8_t> span() const
    {
        return std::visit([](auto& buffer) {
            return buffer ? buffer->span() : std::span<const uint8_t> { };
        }, m_variant);
    }
    std::span<uint8_t> mutableSpan()
    {
        return std::visit([](auto& buffer) {
            return buffer ? buffer->mutableSpan() : std::span<uint8_t> { };
        }, m_variant);
    }

private:
    VariantType m_variant;
};

inline BufferSource toBufferSource(std::span<const uint8_t> data)
{
    return BufferSource(JSC::ArrayBuffer::tryCreate(data));
}

#if PLATFORM(COCOA) && defined(__OBJC__)
inline BufferSource toBufferSource(NSData *data)
{
    return BufferSource(JSC::ArrayBuffer::tryCreate(span(data)));
}

inline RetainPtr<NSData> toNSData(const BufferSource& data)
{
    return WTF::toNSData(data.span());
}
#endif

} // namespace WebCore

#if PLATFORM(COCOA) && defined(__OBJC__)
using WebCore::toNSData;
#endif
