/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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

#if HAVE(HDR_SUPPORT)

#include "PixelBuffer.h"
#include <JavaScriptCore/Float16Array.h>

namespace WebCore {

class Float16ArrayPixelBuffer : public PixelBuffer {
public:
    WEBCORE_EXPORT static Ref<Float16ArrayPixelBuffer> create(const PixelBufferFormat&, const IntSize&, JSC::Float16Array&);
    WEBCORE_EXPORT static std::optional<Ref<Float16ArrayPixelBuffer>> create(const PixelBufferFormat&, const IntSize&, std::span<const Float16> data);

    WEBCORE_EXPORT static RefPtr<Float16ArrayPixelBuffer> tryCreate(const PixelBufferFormat&, const IntSize&);
    WEBCORE_EXPORT static RefPtr<Float16ArrayPixelBuffer> tryCreate(const PixelBufferFormat&, const IntSize&, Ref<JSC::ArrayBuffer>&&);

    JSC::Float16Array& data() const { return m_data.get(); }
    Ref<JSC::Float16Array>&& takeData() { return WTFMove(m_data); }
    WEBCORE_EXPORT std::span<const uint8_t> span() const;

    Type type() const override { return Type::Float16Array; }
    RefPtr<PixelBuffer> createScratchPixelBuffer(const IntSize&) const override;

private:
    Float16ArrayPixelBuffer(const PixelBufferFormat&, const IntSize&, Ref<JSC::Float16Array>&&);

    Ref<JSC::Float16Array> m_data;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Float16ArrayPixelBuffer)
    static bool isType(const WebCore::PixelBuffer& pixelBuffer) { return pixelBuffer.type() == WebCore::PixelBuffer::Type::Float16Array; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // HAVE(HDR_SUPPORT)
