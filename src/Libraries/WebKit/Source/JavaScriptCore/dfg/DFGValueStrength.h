/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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

#if ENABLE(DFG_JIT)

#include <wtf/PrintStream.h>

namespace JSC { namespace DFG {

enum ValueStrength {
    // The value has been used for optimization and it arose through inference. We don't want the
    // fact that we optimized the code to result in the GC keeping this value alive unnecessarily,
    // so we'd rather kill the code and recompile than keep the object alive longer.
    WeakValue,
    
    // The code will keep this value alive. This is true of constants that were present in the
    // source. String constants tend to be strong.
    StrongValue
};

inline ValueStrength merge(ValueStrength a, ValueStrength b)
{
    switch (a) {
    case WeakValue:
        return b;
    case StrongValue:
        return StrongValue;
    }
    RELEASE_ASSERT_NOT_REACHED();

    return WeakValue;
}

} } // namespace JSC::DFG

namespace WTF {

void printInternal(PrintStream&, JSC::DFG::ValueStrength);

} // namespace WTF

#endif // ENABLE(DFG_JIT)
