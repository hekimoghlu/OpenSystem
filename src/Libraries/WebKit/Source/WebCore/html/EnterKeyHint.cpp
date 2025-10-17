/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 31, 2024.
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
#include "EnterKeyHint.h"

#include "CommonAtomStrings.h"
#include <wtf/SortedArrayMap.h>

namespace WebCore {

EnterKeyHint enterKeyHintForAttributeValue(StringView value)
{
    static constexpr std::pair<PackedLettersLiteral<uint64_t>, EnterKeyHint> mappings[] = {
        { "done"_s, EnterKeyHint::Done },
        { "enter"_s, EnterKeyHint::Enter },
        { "go"_s, EnterKeyHint::Go },
        { "next"_s, EnterKeyHint::Next },
        { "previous"_s, EnterKeyHint::Previous },
        { "search"_s, EnterKeyHint::Search },
        { "send"_s, EnterKeyHint::Send }
    };
    static constexpr SortedArrayMap enterKeyHints { mappings };
    return enterKeyHints.get(value, EnterKeyHint::Unspecified);
}

String attributeValueForEnterKeyHint(EnterKeyHint hint)
{
    switch (hint) {
    case EnterKeyHint::Unspecified:
        return emptyAtom();
    case EnterKeyHint::Enter:
        return "enter"_s;
    case EnterKeyHint::Done:
        return "done"_s;
    case EnterKeyHint::Go:
        return "go"_s;
    case EnterKeyHint::Next:
        return "next"_s;
    case EnterKeyHint::Previous:
        return "previous"_s;
    case EnterKeyHint::Search:
        return searchAtom();
    case EnterKeyHint::Send:
        return "send"_s;
    }
    ASSERT_NOT_REACHED();
    return nullAtom();
}

} // namespace WebCore
