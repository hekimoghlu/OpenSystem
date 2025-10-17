/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#include <unicode/utf8.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/LChar.h>

namespace PAL {

class TextCodecUTF8 final : public TextCodec {
    WTF_MAKE_TZONE_ALLOCATED(TextCodecUTF8);
public:
    static void registerEncodingNames(EncodingNameRegistrar);
    static void registerCodecs(TextCodecRegistrar);

    static Vector<uint8_t> encodeUTF8(StringView);
    static std::unique_ptr<TextCodecUTF8> codec();

private:
    void stripByteOrderMark() final { m_shouldStripByteOrderMark = true; }
    String decode(std::span<const uint8_t>, bool flush, bool stopOnError, bool& sawError) final;
    Vector<uint8_t> encode(StringView, UnencodableHandling) const final;

    bool handlePartialSequence(std::span<LChar>& destination, std::span<const uint8_t>& source, bool flush);
    void handlePartialSequence(std::span<UChar>& destination, std::span<const uint8_t>& source, bool flush, bool stopOnError, bool& sawError);
    void consumePartialSequenceByte();

    int m_partialSequenceSize { 0 };
    std::array<uint8_t, U8_MAX_LENGTH> m_partialSequence;
    bool m_shouldStripByteOrderMark { false };
};

} // namespace PAL
