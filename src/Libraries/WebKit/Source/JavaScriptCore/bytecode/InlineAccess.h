/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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

#if ENABLE(JIT)

#include "CodeLocation.h"
#include "PropertyOffset.h"

namespace JSC {

class CodeBlock;
class InlineCacheHandler;
class JSArray;
class Structure;
class StructureStubInfo;
class VM;

class InlineAccess {
public:

    // This is the maximum between inline and out of line self access cases.
    static constexpr size_t sizeForPropertyAccess()
    {
#if CPU(X86_64)
        return 26;
#elif CPU(ARM64)
        return 40;
#elif CPU(ARM_THUMB2)
        return 48;
#elif CPU(RISCV64)
        return 44;
#else
#error "unsupported platform"
#endif
    }

    // This is the maximum between inline and out of line property replace cases.
    static constexpr size_t sizeForPropertyReplace()
    {
#if CPU(X86_64)
        return 26;
#elif CPU(ARM64)
        return 40;
#elif CPU(ARM_THUMB2)
        return 48;
#elif CPU(RISCV64)
        return 52;
#else
#error "unsupported platform"
#endif
    }

    // This is the maximum between array length, string length, and regular self access sizes.
    ALWAYS_INLINE static constexpr size_t sizeForLengthAccess()
    {
#if CPU(X86_64)
        size_t size = 43;
#elif CPU(ARM64)
        size_t size = 44;
#elif CPU(ARM_THUMB2)
        size_t size = 30;
#elif CPU(RISCV64)
        size_t size = 60;
#else
#error "unsupported platform"
#endif
        return std::max(size, sizeForPropertyAccess());
    }

    static bool generateSelfPropertyAccess(StructureStubInfo&, Structure*, PropertyOffset);
    static bool canGenerateSelfPropertyReplace(StructureStubInfo&, PropertyOffset);
    static bool generateSelfPropertyReplace(StructureStubInfo&, Structure*, PropertyOffset);
    static bool isCacheableArrayLength(StructureStubInfo&, JSArray*);
    static bool isCacheableStringLength(StructureStubInfo&);
    static bool generateArrayLength(StructureStubInfo&, JSArray*);
    static bool generateSelfInAccess(StructureStubInfo&, Structure*);
    static bool generateStringLength(StructureStubInfo&);

    // This is helpful when determining the size of an IC on
    // various platforms. When adding a new type of IC, implement
    // its placeholder code here, and log the size. That way we
    // can intelligently choose sizes on various platforms.
    JS_EXPORT_PRIVATE NO_RETURN_DUE_TO_CRASH static void dumpCacheSizesAndCrash();
};

} // namespace JSC

#endif // ENABLE(JIT)
