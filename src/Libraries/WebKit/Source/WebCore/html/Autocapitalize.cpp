/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
#include "Autocapitalize.h"

#if ENABLE(AUTOCAPITALIZE)

#include "CommonAtomStrings.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

AutocapitalizeType autocapitalizeTypeForAttributeValue(const AtomString& attributeValue)
{
    // Omitted / missing values are the Default state.
    if (attributeValue.isEmpty())
        return AutocapitalizeType::Default;

    if (equalLettersIgnoringASCIICase(attributeValue, "on"_s) || equalLettersIgnoringASCIICase(attributeValue, "sentences"_s))
        return AutocapitalizeType::Sentences;
    if (equalLettersIgnoringASCIICase(attributeValue, "off"_s) || equalLettersIgnoringASCIICase(attributeValue, "none"_s))
        return AutocapitalizeType::None;
    if (equalLettersIgnoringASCIICase(attributeValue, "words"_s))
        return AutocapitalizeType::Words;
    if (equalLettersIgnoringASCIICase(attributeValue, "characters"_s))
        return AutocapitalizeType::AllCharacters;

    // Unrecognized values fall back to "on".
    return AutocapitalizeType::Sentences;
}

const AtomString& stringForAutocapitalizeType(AutocapitalizeType type)
{
    switch (type) {
    case AutocapitalizeType::Default:
        return nullAtom();
    case AutocapitalizeType::None:
        return noneAtom();
    case AutocapitalizeType::Sentences: {
        static MainThreadNeverDestroyed<const AtomString> valueSentences("sentences"_s);
        return valueSentences;
    }
    case AutocapitalizeType::Words: {
        static MainThreadNeverDestroyed<const AtomString> valueWords("words"_s);
        return valueWords;
    }
    case AutocapitalizeType::AllCharacters: {
        static MainThreadNeverDestroyed<const AtomString> valueAllCharacters("characters"_s);
        return valueAllCharacters;
    }
    }

    ASSERT_NOT_REACHED();
    return nullAtom();
}

} // namespace WebCore

#endif // ENABLE(AUTOCAPITALIZE)
