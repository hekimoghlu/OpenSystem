/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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

#include "TextCodec.h"
#include <optional>
#include <wtf/TZoneMalloc.h>

namespace PAL {

class TextCodecUTF16 final : public TextCodec {
    WTF_MAKE_TZONE_ALLOCATED(TextCodecUTF16);
public:
    static void registerEncodingNames(EncodingNameRegistrar);
    static void registerCodecs(TextCodecRegistrar);

    explicit TextCodecUTF16(bool littleEndian);

private:
    void stripByteOrderMark() final { m_shouldStripByteOrderMark = true; }
    String decode(std::span<const uint8_t>, bool flush, bool stopOnError, bool& sawError) final;
    Vector<uint8_t> encode(StringView, UnencodableHandling) const final;

    bool m_littleEndian;
    std::optional<uint8_t> m_leadByte;
    std::optional<UChar> m_leadSurrogate;
    bool m_shouldStripByteOrderMark { false };
};

} // namespace PAL
