/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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

#include "UnencodableHandling.h"
#include <array>
#include <memory>
#include <span>
#include <unicode/umachine.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace PAL {

class TextEncoding;

using UnencodableReplacementArray = std::array<char, 32>;

class TextCodec {
    WTF_MAKE_TZONE_ALLOCATED(TextCodec);
    WTF_MAKE_NONCOPYABLE(TextCodec);
public:
    TextCodec() = default;
    virtual ~TextCodec() = default;

    virtual void stripByteOrderMark() { }
    virtual String decode(std::span<const uint8_t> data, bool flush, bool stopOnError, bool& sawError) = 0;

    virtual Vector<uint8_t> encode(StringView, UnencodableHandling) const = 0;

    // Fills a null-terminated string representation of the given
    // unencodable character into the given replacement buffer.
    // The length of the string (not including the null) will be returned.
    static std::span<char> getUnencodableReplacement(char32_t, UnencodableHandling, UnencodableReplacementArray& LIFETIME_BOUND);
};

Function<void(char32_t, Vector<uint8_t>&)> unencodableHandler(UnencodableHandling);

using EncodingNameRegistrar = void (*)(ASCIILiteral alias, ASCIILiteral name);

using NewTextCodecFunction = Function<std::unique_ptr<TextCodec>()>;
using TextCodecRegistrar = void (*)(ASCIILiteral name, NewTextCodecFunction&&);

} // namespace PAL
