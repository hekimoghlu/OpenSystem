/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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
#include "ErrorType.h"

#include <wtf/PrintStream.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

ASCIILiteral errorTypeName(ErrorType errorType)
{
    return errorTypeName(static_cast<ErrorTypeWithExtension>(errorType));
}

ASCIILiteral errorTypeName(ErrorTypeWithExtension errorType)
{
    static const ASCIILiteral errorTypeNames[] = {
#define DECLARE_ERROR_TYPES_STRING(name) #name ""_s,
        JSC_ERROR_TYPES_WITH_EXTENSION(DECLARE_ERROR_TYPES_STRING)
#undef DECLARE_ERROR_TYPES_STRING
    };
    return errorTypeNames[static_cast<unsigned>(errorType)];
}

} // namespace JSC

namespace WTF {

void printInternal(PrintStream& out, JSC::ErrorType errorType)
{
    out.print(JSC::errorTypeName(errorType));
}

void printInternal(PrintStream& out, JSC::ErrorTypeWithExtension errorType)
{
    out.print(JSC::errorTypeName(errorType));
}

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
