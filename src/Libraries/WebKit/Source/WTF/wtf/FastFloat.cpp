/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
#include "config.h"
#include "FastFloat.h"

#include "fast_float/fast_float.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

double parseDouble(std::span<const LChar> string, size_t& parsedLength)
{
    double doubleValue = 0;
    auto stringData = byteCast<char>(string.data());
    auto result = fast_float::from_chars(stringData, stringData + string.size(), doubleValue);
    parsedLength = result.ptr - stringData;
    return doubleValue;
}

double parseDouble(std::span<const UChar> string, size_t& parsedLength)
{
    double doubleValue = 0;
    auto stringData = reinterpret_cast<const char16_t*>(string.data());
    auto result = fast_float::from_chars(stringData, stringData + string.size(), doubleValue);
    parsedLength = result.ptr - stringData;
    return doubleValue;
}

double parseHexDouble(std::span<const LChar> string, size_t& parsedLength)
{
    double doubleValue = 0;
    auto stringData = byteCast<char>(string.data());
    auto result = fast_float::from_chars(stringData, stringData + string.size(), doubleValue, fast_float::chars_format::hex);
    parsedLength = result.ptr - stringData;
    return doubleValue;
}

double parseHexDouble(std::span<const UChar> string, size_t& parsedLength)
{
    double doubleValue = 0;
    auto stringData = reinterpret_cast<const char16_t*>(string.data());
    auto result = fast_float::from_chars(stringData, stringData + string.size(), doubleValue, fast_float::chars_format::hex);
    parsedLength = result.ptr - stringData;
    return doubleValue;
}

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
