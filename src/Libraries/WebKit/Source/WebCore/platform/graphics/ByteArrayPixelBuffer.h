/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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

#include "PixelBuffer.h"
#include <JavaScriptCore/Uint8ClampedArray.h>

namespace WebCore {

class ByteArrayPixelBuffer : public PixelBuffer {
public:
    WEBCORE_EXPORT static Ref<ByteArrayPixelBuffer> create(const PixelBufferFormat&, const IntSize&, JSC::Uint8ClampedArray&);
    WEBCORE_EXPORT static std::optional<Ref<ByteArrayPixelBuffer>> create(const PixelBufferFormat&, const IntSize&, std::span<const uint8_t> data);

    WEBCORE_EXPORT static RefPtr<ByteArrayPixelBuffer> tryCreate(const PixelBufferFormat&, const IntSize&);
    WEBCORE_EXPORT static RefPtr<ByteArrayPixelBuffer> tryCreate(const PixelBufferFormat&, const IntSize&, Ref<JSC::ArrayBuffer>&&);

    JSC::Uint8ClampedArray& data() const { return m_data.get(); }
    Ref<JSC::Uint8ClampedArray>&& takeData() { return WTFMove(m_data); }
    WEBCORE_EXPORT std::span<const uint8_t> span() const;

    Type type() const override { return Type::ByteArray; }
    RefPtr<PixelBuffer> createScratchPixelBuffer(const IntSize&) const override;

private:
    ByteArrayPixelBuffer(const PixelBufferFormat&, const IntSize&, Ref<JSC::Uint8ClampedArray>&&);

    Ref<JSC::Uint8ClampedArray> m_data;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ByteArrayPixelBuffer)
    static bool isType(const WebCore::PixelBuffer& pixelBuffer) { return pixelBuffer.type() == WebCore::PixelBuffer::Type::ByteArray; }
SPECIALIZE_TYPE_TRAITS_END()
