/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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
#include "WGSLEnums.h"

#include <wtf/PrintStream.h>
#include <wtf/SortedArrayMap.h>

namespace WGSL {

#define COMMA ,
#define LPAREN (
#define RPAREN )
#define CONTINUATION(args...) args RPAREN
#define EXPAND(x) x

#define ENUM_DEFINE_TO_STRING_CASE_(__type, __name, __string, ...) \
    case __type::__name: \
        return #__string##_s; \
        break;

#define ENUM_DEFINE_TO_STRING_CASE(__name) \
    ENUM_DEFINE_TO_STRING_CASE_ LPAREN __name, CONTINUATION

#define ENUM_DEFINE_TO_STRING(__name) \
    ASCIILiteral toString(__name __value) \
    { \
        switch (__value) { \
            EXPAND(ENUM_##__name(ENUM_DEFINE_TO_STRING_CASE LPAREN __name RPAREN)) \
        } \
    }

#define ENUM_DEFINE_PRINT_INTERNAL(__name) \
    void printInternal(PrintStream& out, __name __value) \
    { \
        out.print(toString(__value)); \
    }

#define ENUM_DEFINE_PARSE_ENTRY_(__type, __name, __string, ...) \
    { #__string##_s, __type::__name },

#define ENUM_DEFINE_PARSE_ENTRY(__name) \
    ENUM_DEFINE_PARSE_ENTRY_ LPAREN __name, CONTINUATION

#define ENUM_DEFINE_PARSE(__name) \
    const __name* parse##__name(const String& __string) \
    { \
        static constexpr std::pair<ComparableASCIILiteral, __name> __entries[] { \
            EXPAND(ENUM_##__name(ENUM_DEFINE_PARSE_ENTRY LPAREN __name RPAREN)) \
        }; \
        static constexpr SortedArrayMap __map { __entries }; \
        return __map.tryGet(__string); \
    }

#define ENUM_DEFINE(__name) \
    ENUM_DEFINE_TO_STRING(__name) \
    ENUM_DEFINE_PRINT_INTERNAL(__name) \
    ENUM_DEFINE_PARSE(__name)

ENUM_DEFINE(AddressSpace);
ENUM_DEFINE(AccessMode);
ENUM_DEFINE(TexelFormat);
ENUM_DEFINE(InterpolationType);
ENUM_DEFINE(InterpolationSampling);
ENUM_DEFINE(ShaderStage);
ENUM_DEFINE(SeverityControl);
ENUM_DEFINE(Builtin);
ENUM_DEFINE(Extension);
ENUM_DEFINE(LanguageFeature);

#undef ENUM_DEFINE
#undef ENUM_DEFINE_PRINT_INTERNAL
#undef ENUM_DEFINE_TO_STRING
#undef ENUM_DEFINE_TO_STRING_CASE
#undef ENUM_DEFINE_TO_STRING_CASE_
#undef EXPAND
#undef CONTINUATION
#undef RPAREN
#undef LPAREN
#undef COMMA

AccessMode defaultAccessModeForAddressSpace(AddressSpace addressSpace)
{
    // Default access mode based on address space
    // https://www.w3.org/TR/WGSL/#address-space
    switch (addressSpace) {
    case AddressSpace::Function:
    case AddressSpace::Private:
    case AddressSpace::Workgroup:
        return AccessMode::ReadWrite;
    case AddressSpace::Uniform:
    case AddressSpace::Storage:
    case AddressSpace::Handle:
        return AccessMode::Read;
    }
}

} // namespace WGSL
