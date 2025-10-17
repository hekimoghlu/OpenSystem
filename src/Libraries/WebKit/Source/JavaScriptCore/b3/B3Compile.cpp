/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
#include "B3Compile.h"

#if ENABLE(B3_JIT)

#include "B3Generate.h"
#include "B3Procedure.h"
#include "CCallHelpers.h"
#include "CompilerTimingScope.h"
#include "LinkBuffer.h"

namespace JSC { namespace B3 {

Compilation compile(Procedure& proc)
{
    CompilerTimingScope timingScope("Total B3+Air"_s, "compile"_s);
    
    prepareForGeneration(proc);
    
    CCallHelpers jit;
    generate(proc, jit);
    LinkBuffer linkBuffer(jit, nullptr);

    return Compilation(FINALIZE_CODE(linkBuffer, JITCompilationPtrTag, nullptr, "Compilation"), proc.releaseByproducts());
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

