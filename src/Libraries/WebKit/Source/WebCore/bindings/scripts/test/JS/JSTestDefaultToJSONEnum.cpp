/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#include "JSTestDefaultToJSONEnum.h"

#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSString.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/SortedArrayMap.h>



namespace WebCore {
using namespace JSC;

String convertEnumerationToString(TestDefaultToJSONEnum enumerationValue)
{
    static const std::array<NeverDestroyed<String>, 2> values {
        MAKE_STATIC_STRING_IMPL("EnumValue1"),
        MAKE_STATIC_STRING_IMPL("EnumValue2"),
    };
    static_assert(static_cast<size_t>(TestDefaultToJSONEnum::EnumValue1) == 0, "TestDefaultToJSONEnum::EnumValue1 is not 0 as expected");
    static_assert(static_cast<size_t>(TestDefaultToJSONEnum::EnumValue2) == 1, "TestDefaultToJSONEnum::EnumValue2 is not 1 as expected");
    ASSERT(static_cast<size_t>(enumerationValue) < std::size(values));
    return values[static_cast<size_t>(enumerationValue)];
}

template<> JSString* convertEnumerationToJS(VM& vm, TestDefaultToJSONEnum enumerationValue)
{
    return jsStringWithCache(vm, convertEnumerationToString(enumerationValue));
}

template<> std::optional<TestDefaultToJSONEnum> parseEnumerationFromString<TestDefaultToJSONEnum>(const String& stringValue)
{
    static constexpr std::array<std::pair<ComparableASCIILiteral, TestDefaultToJSONEnum>, 2> mappings {
        std::pair<ComparableASCIILiteral, TestDefaultToJSONEnum> { "EnumValue1"_s, TestDefaultToJSONEnum::EnumValue1 },
        std::pair<ComparableASCIILiteral, TestDefaultToJSONEnum> { "EnumValue2"_s, TestDefaultToJSONEnum::EnumValue2 },
    };
    static constexpr SortedArrayMap enumerationMapping { mappings };
    if (auto* enumerationValue = enumerationMapping.tryGet(stringValue); LIKELY(enumerationValue))
        return *enumerationValue;
    return std::nullopt;
}

template<> std::optional<TestDefaultToJSONEnum> parseEnumeration<TestDefaultToJSONEnum>(JSGlobalObject& lexicalGlobalObject, JSValue value)
{
    return parseEnumerationFromString<TestDefaultToJSONEnum>(value.toWTFString(&lexicalGlobalObject));
}

template<> ASCIILiteral expectedEnumerationValues<TestDefaultToJSONEnum>()
{
    return "\"EnumValue1\", \"EnumValue2\""_s;
}

} // namespace WebCore

