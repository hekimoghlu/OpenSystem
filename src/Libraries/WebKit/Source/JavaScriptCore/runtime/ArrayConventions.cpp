/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
#include "ArrayConventions.h"

#include "GCMemoryOperations.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

#if USE(JSVALUE64)
void clearArrayMemset(WriteBarrier<Unknown>* base, unsigned count)
{
    gcSafeZeroMemory(base, count * sizeof(WriteBarrier<Unknown>));
}

void clearArrayMemset(double* base, unsigned count)
{
#if CPU(X86_64)
    uint64_t pnan = std::bit_cast<uint64_t>(PNaN);
    asm volatile (
        "rep stosq\n\t"
        : "+D"(base), "+c"(count)
        : "a"(pnan)
        : "memory"
        );
#else // not CPU(X86_64)
    // Oh no, we can't actually do any better than this!
    for (unsigned i = count; i--;)
        base[i] = PNaN;
#endif // generic CPU
}
#endif // USE(JSVALUE64)

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
