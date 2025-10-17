/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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
#include "YarrUnicodeProperties.h"

#include "Yarr.h"
#include "YarrPattern.h"
#include <string_view>
#include <wtf/text/WTFString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace Yarr {

struct HashIndex {
    int16_t value;
    int16_t next;
};

struct HashValue {
    const char* key;
    int index;
};

struct HashTable {
    int numberOfValues;
    int indexMask;
    const HashValue* values;
    const HashIndex* index;

    ALWAYS_INLINE int entry(WTF::String& key) const
    {
        int indexEntry = key.impl()->hash() & indexMask;
        int valueIndex = index[indexEntry].value;

        if (valueIndex == -1)
            return -1;

        while (true) {
            if (WTF::equal(key, StringView::fromLatin1(values[valueIndex].key)))
                return values[valueIndex].index;

            indexEntry = index[indexEntry].next;
            if (indexEntry == -1)
                return -1;
            valueIndex = index[indexEntry].value;
            ASSERT(valueIndex != -1);
        };
    }
};

#include "UnicodePatternTables.h"

std::optional<BuiltInCharacterClassID> unicodeMatchPropertyValue(WTF::String unicodePropertyName, WTF::String unicodePropertyValue)
{
    int propertyIndex = -1;

    if (unicodePropertyName == "Script"_s || unicodePropertyName == "sc"_s)
        propertyIndex = scriptHashTable.entry(unicodePropertyValue);
    else if (unicodePropertyName == "Script_Extensions"_s || unicodePropertyName == "scx"_s)
        propertyIndex = scriptExtensionHashTable.entry(unicodePropertyValue);
    else if (unicodePropertyName == "General_Category"_s || unicodePropertyName == "gc"_s)
        propertyIndex = generalCategoryHashTable.entry(unicodePropertyValue);

    if (propertyIndex == -1)
        return std::nullopt;

    return std::optional<BuiltInCharacterClassID>(static_cast<BuiltInCharacterClassID>(static_cast<int>(BuiltInCharacterClassID::BaseUnicodePropertyID) + propertyIndex));
}

std::optional<BuiltInCharacterClassID> unicodeMatchProperty(WTF::String unicodePropertyValue, CompileMode compileMode)
{
    int propertyIndex = -1;

    propertyIndex = binaryPropertyHashTable.entry(unicodePropertyValue);
    if (propertyIndex == -1)
        propertyIndex = generalCategoryHashTable.entry(unicodePropertyValue);
    if (propertyIndex == -1 && compileMode == CompileMode::UnicodeSets)
        propertyIndex = sequencePropertyHashTable.entry(unicodePropertyValue);

    if (propertyIndex == -1)
        return std::nullopt;

    return std::optional<BuiltInCharacterClassID>(static_cast<BuiltInCharacterClassID>(static_cast<int>(BuiltInCharacterClassID::BaseUnicodePropertyID) + propertyIndex));
}

std::unique_ptr<CharacterClass> createUnicodeCharacterClassFor(BuiltInCharacterClassID unicodeClassID)
{
    unsigned unicodePropertyIndex = static_cast<unsigned>(unicodeClassID) - static_cast<unsigned>(BuiltInCharacterClassID::BaseUnicodePropertyID);

    return createCharacterClassFunctions[unicodePropertyIndex]();
}

bool characterClassMayContainStrings(BuiltInCharacterClassID unicodeClassID)
{
    unsigned unicodePropertyIndex = static_cast<unsigned>(unicodeClassID) - static_cast<unsigned>(BuiltInCharacterClassID::BaseUnicodePropertyID);

    return unicodeCharacterClassMayContainStrings(unicodePropertyIndex);
}

} } // namespace JSC::Yarr

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
