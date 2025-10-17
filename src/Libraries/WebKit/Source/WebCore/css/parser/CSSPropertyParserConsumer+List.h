/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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

#include "CSSParserTokenRange.h"
#include "CSSPropertyParserConsumer+Primitives.h"
#include "CSSValueList.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

template<typename SubConsumer, typename... Args>
RefPtr<CSSValue> consumeCommaSeparatedListWithSingleValueOptimization(CSSParserTokenRange& range, SubConsumer&& subConsumer, Args&&... args)
{
    CSSValueListBuilder list;
    do {
        auto value = std::invoke(subConsumer, range, std::forward<Args>(args)...);
        if (!value)
            return nullptr;
        list.append(value.releaseNonNull());
    } while (consumeCommaIncludingWhitespace(range));
    if (list.size() == 1)
        return WTFMove(list[0]);
    return CSSValueList::createCommaSeparated(WTFMove(list));
}

template<typename SubConsumer, typename... Args>
RefPtr<CSSValueList> consumeCommaSeparatedListWithoutSingleValueOptimization(CSSParserTokenRange& range, SubConsumer&& subConsumer, Args&&... args)
{
    CSSValueListBuilder list;
    do {
        auto value = std::invoke(subConsumer, range, std::forward<Args>(args)...);
        if (!value)
            return nullptr;
        list.append(value.releaseNonNull());
    } while (consumeCommaIncludingWhitespace(range));
    return CSSValueList::createCommaSeparated(WTFMove(list));
}

} // namespace CSSPropertyParserHelpers

} // namespace WebCore
