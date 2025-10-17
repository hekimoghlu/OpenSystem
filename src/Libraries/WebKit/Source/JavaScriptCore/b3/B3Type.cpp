/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#include "B3Type.h"

#if ENABLE(B3_JIT)

#include <wtf/PrintStream.h>

namespace WTF {

using namespace JSC::B3;

void printInternal(PrintStream& out, Type type)
{
    switch (type.kind()) {
    case Void:
        out.print("Void");
        return;
    case Int32:
        out.print("Int32");
        return;
    case Int64:
        out.print("Int64");
        return;
    case Float:
        out.print("Float");
        return;
    case Double:
        out.print("Double");
        return;
    case V128:
        out.print("V128");
        return;
    case Tuple:
        out.print("Tuple");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

static_assert(std::is_standard_layout_v<JSC::B3::TypeKind> && std::is_trivial_v<JSC::B3::TypeKind>);
} // namespace WTF


#endif // ENABLE(B3_JIT)
