/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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

#if ENABLE(CONDITION)

#include "JSTestStandaloneEnumeration.h"

#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSString.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/SortedArrayMap.h>



namespace WebCore {
using namespace JSC;

String convertEnumerationToString(TestStandaloneEnumeration enumerationValue)
{
    static const std::array<NeverDestroyed<String>, 2> values {
        MAKE_STATIC_STRING_IMPL("enumValue1"),
        MAKE_STATIC_STRING_IMPL("enumValue2"),
    };
    static_assert(static_cast<size_t>(TestStandaloneEnumeration::EnumValue1) == 0, "TestStandaloneEnumeration::EnumValue1 is not 0 as expected");
    static_assert(static_cast<size_t>(TestStandaloneEnumeration::EnumValue2) == 1, "TestStandaloneEnumeration::EnumValue2 is not 1 as expected");
    ASSERT(static_cast<size_t>(enumerationValue) < std::size(values));
    return values[static_cast<size_t>(enumerationValue)];
}

template<> JSString* convertEnumerationToJS(VM& vm, TestStandaloneEnumeration enumerationValue)
{
    return jsStringWithCache(vm, convertEnumerationToString(enumerationValue));
}

template<> std::optional<TestStandaloneEnumeration> parseEnumerationFromString<TestStandaloneEnumeration>(const String& stringValue)
{
    static constexpr std::array<std::pair<ComparableASCIILiteral, TestStandaloneEnumeration>, 2> mappings {
        std::pair<ComparableASCIILiteral, TestStandaloneEnumeration> { "enumValue1"_s, TestStandaloneEnumeration::EnumValue1 },
        std::pair<ComparableASCIILiteral, TestStandaloneEnumeration> { "enumValue2"_s, TestStandaloneEnumeration::EnumValue2 },
    };
    static constexpr SortedArrayMap enumerationMapping { mappings };
    if (auto* enumerationValue = enumerationMapping.tryGet(stringValue); LIKELY(enumerationValue))
        return *enumerationValue;
    return std::nullopt;
}

template<> std::optional<TestStandaloneEnumeration> parseEnumeration<TestStandaloneEnumeration>(JSGlobalObject& lexicalGlobalObject, JSValue value)
{
    return parseEnumerationFromString<TestStandaloneEnumeration>(value.toWTFString(&lexicalGlobalObject));
}

template<> ASCIILiteral expectedEnumerationValues<TestStandaloneEnumeration>()
{
    return "\"enumValue1\", \"enumValue2\""_s;
}

} // namespace WebCore


#endif // ENABLE(CONDITION)
