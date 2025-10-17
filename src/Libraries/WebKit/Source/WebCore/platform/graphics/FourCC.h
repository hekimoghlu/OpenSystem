/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

struct FourCC {
    constexpr FourCC() = default;
    constexpr FourCC(uint32_t value) : value { value } { }
    constexpr FourCC(std::span<const char, 5> nullTerminatedString);
    constexpr std::array<char, 5> string() const;
    static std::optional<FourCC> fromString(StringView);
    friend constexpr bool operator==(FourCC, FourCC) = default;

    uint32_t value { 0 };
};

constexpr FourCC::FourCC(std::span<const char, 5> data)
    : value(data[0] << 24 | data[1] << 16 | data[2] << 8 | data[3])
{
    ASSERT_UNDER_CONSTEXPR_CONTEXT(isASCII(data[0]));
    ASSERT_UNDER_CONSTEXPR_CONTEXT(isASCII(data[1]));
    ASSERT_UNDER_CONSTEXPR_CONTEXT(isASCII(data[2]));
    ASSERT_UNDER_CONSTEXPR_CONTEXT(isASCII(data[3]));
    ASSERT_UNDER_CONSTEXPR_CONTEXT(data[4] == '\0');
}

constexpr std::array<char, 5> FourCC::string() const
{
    return {
        static_cast<char>(value >> 24),
        static_cast<char>(value >> 16),
        static_cast<char>(value >> 8),
        static_cast<char>(value),
        '\0'
    };
}

} // namespace WebCore

namespace WTF {

template<typename> struct LogArgument;

template<> struct LogArgument<WebCore::FourCC> {
    static String toString(const WebCore::FourCC& code) { return String::fromLatin1(code.string().data()); }
};

} // namespace WTF
