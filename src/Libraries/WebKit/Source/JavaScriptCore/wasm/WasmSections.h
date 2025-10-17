/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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

#if ENABLE(WEBASSEMBLY)

#if !ASSERT_ENABLED
IGNORE_RETURN_TYPE_WARNINGS_BEGIN
#endif

#include <wtf/text/ASCIILiteral.h>

namespace JSC { namespace Wasm {

// macro(Name, ID, OrderingNumber, Description).
#define FOR_EACH_KNOWN_WASM_SECTION(macro) \
    macro(Type,       1,  1, "Function signature declarations") \
    macro(Import,     2,  2, "Import declarations") \
    macro(Function,   3,  3, "Function declarations") \
    macro(Table,      4,  4, "Indirect function table and other tables") \
    macro(Memory,     5,  5, "Memory attributes") \
    macro(Global,     6,  7, "Global declarations") \
    macro(Export,     7,  8, "Exports") \
    macro(Start,      8,  9, "Start function declaration") \
    macro(Element,    9, 10, "Elements section") \
    macro(Code,      10, 12, "Function bodies (code)") \
    macro(Data,      11, 13, "Data segments") \
    macro(DataCount, 12, 11, "Data count") \
    macro(Exception, 13,  6, "Exception declarations") \

enum class Section : uint8_t {
    // It's important that Begin is less than every other section number and that Custom is greater.
    // This only works because section numbers are currently monotonically increasing.
    // Also, Begin is not a real section but is used as a marker for validating the ordering
    // of sections.
    Begin = 0,
#define DEFINE_WASM_SECTION_ENUM(NAME, ID, ORDERING, DESCRIPTION) NAME = ID,
    FOR_EACH_KNOWN_WASM_SECTION(DEFINE_WASM_SECTION_ENUM)
#undef DEFINE_WASM_SECTION_ENUM
    Custom
};
static_assert(static_cast<uint8_t>(Section::Begin) < static_cast<uint8_t>(Section::Type), "Begin should come before the first known section.");

inline unsigned orderingNumber(Section section)
{
    switch (section) {
#define ORDERING_OF_SECTION(NAME, ID, ORDERING, DESCRIPTION) case Section::NAME: return ORDERING;
        FOR_EACH_KNOWN_WASM_SECTION(ORDERING_OF_SECTION)
#undef VALIDATE_SECTION
    default:
        return static_cast<unsigned>(section);
    }
}

template<typename Int>
inline bool isKnownSection(Int section)
{
    switch (section) {
#define VALIDATE_SECTION(NAME, ID, ORDERING, DESCRIPTION) case static_cast<Int>(Section::NAME): return true;
        FOR_EACH_KNOWN_WASM_SECTION(VALIDATE_SECTION)
#undef VALIDATE_SECTION
    default:
        return false;
    }
}

inline bool decodeSection(uint8_t sectionByte, Section& section)
{
    section = Section::Custom;
    if (!sectionByte)
        return true;

    if (!isKnownSection(sectionByte))
        return false;

    section = static_cast<Section>(sectionByte);
    return true;
}

inline bool validateOrder(Section previousKnown, Section next)
{
    ASSERT(isKnownSection(previousKnown) || previousKnown == Section::Begin);
    return orderingNumber(previousKnown) < orderingNumber(next);
}

inline ASCIILiteral makeString(Section section)
{
    switch (section) {
    case Section::Begin:
        return "Begin"_s;
    case Section::Custom:
        return "Custom"_s;
#define STRINGIFY_SECTION_NAME(NAME, ID, ORDERING, DESCRIPTION) case Section::NAME: return #NAME ## _s;
        FOR_EACH_KNOWN_WASM_SECTION(STRINGIFY_SECTION_NAME)
#undef STRINGIFY_SECTION_NAME
    }
    ASSERT_NOT_REACHED();
}

} } // namespace JSC::Wasm

#if !ASSERT_ENABLED
IGNORE_RETURN_TYPE_WARNINGS_END
#endif

#endif // ENABLE(WEBASSEMBLY)
