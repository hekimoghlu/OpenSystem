/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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

#include "Operands.h"
#include <wtf/CommaPrinter.h>

namespace JSC {

inline void Operand::dump(PrintStream& out) const
{
    if (isTmp())
        out.print("tmp", value());
    else
        out.print(virtualRegister());
}

template<typename T, typename U>
void Operands<T, U>::dumpInContext(PrintStream& out, DumpContext* context) const
{
    CommaPrinter comma(" "_s);
    for (size_t argumentIndex = numberOfArguments(); argumentIndex--;) {
        if (!argument(argumentIndex))
            continue;
        out.print(comma, "arg"_s, argumentIndex, ":"_s, inContext(argument(argumentIndex), context));
    }
    for (size_t localIndex = 0; localIndex < numberOfLocals(); ++localIndex) {
        if (!local(localIndex))
            continue;
        out.print(comma, "loc"_s, localIndex, ":"_s, inContext(local(localIndex), context));
    }
    for (size_t tmpIndex = 0; tmpIndex < numberOfTmps(); ++tmpIndex) {
        if (!tmp(tmpIndex))
            continue;
        out.print(comma, "tmp"_s, tmpIndex, ":"_s, inContext(tmp(tmpIndex), context));
    }
}

template<typename T, typename U>
void Operands<T, U>::dump(PrintStream& out) const
{
    CommaPrinter comma(" "_s);
    for (size_t argumentIndex = numberOfArguments(); argumentIndex--;) {
        if (!argument(argumentIndex))
            continue;
        out.print(comma, "arg"_s, argumentIndex, ":"_s, argument(argumentIndex));
    }
    for (size_t localIndex = 0; localIndex < numberOfLocals(); ++localIndex) {
        if (!local(localIndex))
            continue;
        out.print(comma, "loc"_s, localIndex, ":"_s, local(localIndex));
    }
    for (size_t tmpIndex = 0; tmpIndex < numberOfTmps(); ++tmpIndex) {
        if (!tmp(tmpIndex))
            continue;
        out.print(comma, "tmp"_s, tmpIndex, ":"_s, tmp(tmpIndex));
    }
}

} // namespace JSC
