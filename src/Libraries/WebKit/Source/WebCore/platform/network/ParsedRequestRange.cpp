/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#include "ParsedRequestRange.h"

#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

std::optional<ParsedRequestRange> ParsedRequestRange::parse(StringView input)
{
    // https://tools.ietf.org/html/rfc7233#section-2.1 but assuming there will always be a begin and an end or parsing will fail
    if (!input.startsWith("bytes="_s))
        return std::nullopt;

    size_t begin { 0 };
    size_t end { 0 };
    size_t rangeBeginPosition = 6;
    size_t dashPosition = input.find('-', rangeBeginPosition);
    if (dashPosition == notFound)
        return std::nullopt;

    auto beginString = input.substring(rangeBeginPosition, dashPosition - rangeBeginPosition);
    auto optionalBegin = parseInteger<uint64_t>(beginString);
    if (!optionalBegin)
        return std::nullopt;
    begin = *optionalBegin;

    auto endString = input.substring(dashPosition + 1);
    auto optionalEnd = parseInteger<uint64_t>(endString);
    if (!optionalEnd)
        return std::nullopt;
    end = *optionalEnd;

    if (begin > end)
        return std::nullopt;

    return {{ begin, end }};
}

}
