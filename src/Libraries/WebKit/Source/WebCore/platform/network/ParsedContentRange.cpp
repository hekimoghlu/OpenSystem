/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#include "ParsedContentRange.h"

#include <wtf/StdLibExtras.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

static bool areContentRangeValuesValid(int64_t firstBytePosition, int64_t lastBytePosition, int64_t instanceLength)
{
    // From <http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html>
    // 14.16 Content-Range
    // A byte-content-range-spec with a byte-range-resp-spec whose last- byte-pos value is less than its first-byte-pos value,
    // or whose instance-length value is less than or equal to its last-byte-pos value, is invalid.
    if (firstBytePosition < 0)
        return false;
    ASSERT(firstBytePosition >= 0);

    if (lastBytePosition < firstBytePosition)
        return false;
    ASSERT(lastBytePosition >= 0);

    if (instanceLength == ParsedContentRange::unknownLength)
        return true;

    return lastBytePosition < instanceLength;
}

static bool parseContentRange(StringView headerValue, int64_t& firstBytePosition, int64_t& lastBytePosition, int64_t& instanceLength)
{
    // From <http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html>
    // 14.16 Content-Range
    //
    // Content-Range = "Content-Range" ":" content-range-spec
    // content-range-spec      = byte-content-range-spec
    // byte-content-range-spec = bytes-unit SP
    //                          byte-range-resp-spec "/"
    //                          ( instance-length | "*" )
    // byte-range-resp-spec = (first-byte-pos "-" last-byte-pos)
    //                               | "*"
    // instance-length           = 1*DIGIT

    static constexpr auto prefix = "bytes "_s;
    static constexpr size_t prefixLength = 6;

    if (!headerValue.startsWith(prefix))
        return false;

    size_t byteSeparatorTokenLoc = headerValue.find('-', prefixLength);
    if (byteSeparatorTokenLoc == notFound)
        return false;

    size_t instanceLengthSeparatorToken = headerValue.find('/', byteSeparatorTokenLoc + 1);
    if (instanceLengthSeparatorToken == notFound)
        return false;

    auto firstByteString = headerValue.substring(prefixLength, byteSeparatorTokenLoc - prefixLength);
    if (!firstByteString.containsOnly<isASCIIDigit>())
        return false;

    auto optionalFirstBytePosition = parseInteger<int64_t>(firstByteString);
    if (!optionalFirstBytePosition)
        return false;
    firstBytePosition = *optionalFirstBytePosition;

    auto lastByteString = headerValue.substring(byteSeparatorTokenLoc + 1, instanceLengthSeparatorToken - (byteSeparatorTokenLoc + 1));
    if (!lastByteString.containsOnly<isASCIIDigit>())
        return false;

    auto optionalLastBytePosition = parseInteger<int64_t>(lastByteString);
    if (!optionalLastBytePosition)
        return false;
    lastBytePosition = *optionalLastBytePosition;

    auto instanceString = headerValue.substring(instanceLengthSeparatorToken + 1);
    if (instanceString == "*"_s)
        instanceLength = ParsedContentRange::unknownLength;
    else {
        if (!instanceString.containsOnly<isASCIIDigit>())
            return false;

        auto optionalInstanceLength = parseInteger<int64_t>(instanceString);
        if (!optionalInstanceLength)
            return false;
        instanceLength = *optionalInstanceLength;
    }

    return areContentRangeValuesValid(firstBytePosition, lastBytePosition, instanceLength);
}

ParsedContentRange::ParsedContentRange(const String& headerValue)
{
    if (!parseContentRange(StringView(headerValue), m_firstBytePosition, m_lastBytePosition, m_instanceLength))
        m_instanceLength = invalidLength;
}

ParsedContentRange::ParsedContentRange(int64_t firstBytePosition, int64_t lastBytePosition, int64_t instanceLength)
    : m_firstBytePosition(firstBytePosition)
    , m_lastBytePosition(lastBytePosition)
    , m_instanceLength(instanceLength)
{
    if (!areContentRangeValuesValid(m_firstBytePosition, m_lastBytePosition, m_instanceLength))
        m_instanceLength = invalidLength;
}

String ParsedContentRange::headerValue() const
{
    if (!isValid())
        return String();
    if (m_instanceLength == unknownLength)
        return makeString("bytes "_s, m_firstBytePosition, '-', m_lastBytePosition, "/*"_s);
    return makeString("bytes "_s, m_firstBytePosition, '-', m_lastBytePosition, '/', m_instanceLength);
}

}
