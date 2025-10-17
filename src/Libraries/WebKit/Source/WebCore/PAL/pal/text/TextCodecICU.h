/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
#include <unicode/ucnv.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/ASCIILiteral.h>
#include <wtf/unicode/icu/ICUHelpers.h>

namespace PAL {

using ICUConverterPtr = std::unique_ptr<UConverter, ICUDeleter<ucnv_close>>;

class TextCodecICU final : public TextCodec {
    WTF_MAKE_TZONE_ALLOCATED(TextCodecICU);
public:
    static void registerEncodingNames(EncodingNameRegistrar);
    static void registerCodecs(TextCodecRegistrar);

    explicit TextCodecICU(ASCIILiteral encoding, ASCIILiteral canonicalConverterName);
    virtual ~TextCodecICU();

private:
    String decode(std::span<const uint8_t>, bool flush, bool stopOnError, bool& sawError) final;
    Vector<uint8_t> encode(StringView, UnencodableHandling) const final;

    void createICUConverter() const;
    void releaseICUConverter() const;

    int decodeToBuffer(std::span<UChar> buffer, std::span<const uint8_t>& source, int32_t* offsets, bool flush, UErrorCode&);

    ASCIILiteral m_encodingName;
    ASCIILiteral const m_canonicalConverterName;
    mutable ICUConverterPtr m_converter;
};

struct ICUConverterWrapper {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    ICUConverterPtr converter;
};

} // namespace PAL
