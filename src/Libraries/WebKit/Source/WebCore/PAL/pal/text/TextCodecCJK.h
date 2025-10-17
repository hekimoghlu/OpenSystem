/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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

class TextCodecCJK final : public TextCodec {
    WTF_MAKE_TZONE_ALLOCATED(TextCodecCJK);
public:
    static void registerEncodingNames(EncodingNameRegistrar);
    static void registerCodecs(TextCodecRegistrar);

    enum class Encoding : uint8_t;
    explicit TextCodecCJK(Encoding);

private:
    String decode(std::span<const uint8_t>, bool flush, bool stopOnError, bool& sawError) final;
    Vector<uint8_t> encode(StringView, UnencodableHandling) const final;

    enum class SawError : bool { No, Yes };
    String decodeCommon(std::span<const uint8_t>, bool, bool, bool&, const Function<SawError(uint8_t, StringBuilder&)>&);

    String eucJPDecode(std::span<const uint8_t>, bool, bool, bool&);
    String iso2022JPDecode(std::span<const uint8_t>, bool, bool, bool&);
    String shiftJISDecode(std::span<const uint8_t>, bool, bool, bool&);
    String eucKRDecode(std::span<const uint8_t>, bool, bool, bool&);
    String big5Decode(std::span<const uint8_t>, bool, bool, bool&);
    String gbkDecode(std::span<const uint8_t>, bool, bool, bool&);
    String gb18030Decode(std::span<const uint8_t>, bool, bool, bool&);

    const Encoding m_encoding;

    bool m_jis0212 { false };

    enum class ISO2022JPDecoderState : uint8_t { ASCII, Roman, Katakana, LeadByte, TrailByte, EscapeStart, Escape };
    ISO2022JPDecoderState m_iso2022JPDecoderState { ISO2022JPDecoderState::ASCII };
    ISO2022JPDecoderState m_iso2022JPDecoderOutputState { ISO2022JPDecoderState::ASCII };
    bool m_iso2022JPOutput { false };
    std::optional<uint8_t> m_iso2022JPSecondPrependedByte;

    uint8_t m_gb18030First { 0x00 };
    uint8_t m_gb18030Second { 0x00 };
    uint8_t m_gb18030Third { 0x00 };

    uint8_t m_lead { 0x00 };
    std::optional<uint8_t> m_prependedByte;
};

} // namespace PAL
