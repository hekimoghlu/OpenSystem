/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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

#include <wtf/text/ASCIILiteral.h>

namespace JSC {

#define JSC_ERROR_TYPES(macro) \
    macro(Error) \
    macro(EvalError) \
    macro(RangeError) \
    macro(ReferenceError) \
    macro(SyntaxError) \
    macro(TypeError) \
    macro(URIError) \
    macro(AggregateError) \

#define JSC_ERROR_TYPES_WITH_EXTENSION(macro) \
    JSC_ERROR_TYPES(macro) \
    macro(OutOfMemoryError) \

enum class ErrorType : uint8_t {
#define DECLARE_ERROR_TYPES_ENUM(name) name,
    JSC_ERROR_TYPES(DECLARE_ERROR_TYPES_ENUM)
#undef DECLARE_ERROR_TYPES_ENUM
};

#define COUNT_ERROR_TYPES(name) 1 +
static constexpr unsigned NumberOfErrorType {
    JSC_ERROR_TYPES(COUNT_ERROR_TYPES) 0
};
#undef COUNT_ERROR_TYPES

enum class ErrorTypeWithExtension : uint8_t {
#define DECLARE_ERROR_TYPES_ENUM(name) name,
    JSC_ERROR_TYPES_WITH_EXTENSION(DECLARE_ERROR_TYPES_ENUM)
#undef DECLARE_ERROR_TYPES_ENUM
};

ASCIILiteral errorTypeName(ErrorType);
ASCIILiteral errorTypeName(ErrorTypeWithExtension);

} // namespace JSC

namespace WTF {

class PrintStream;

void printInternal(PrintStream&, JSC::ErrorType);
void printInternal(PrintStream&, JSC::ErrorTypeWithExtension);

} // namespace WTF
