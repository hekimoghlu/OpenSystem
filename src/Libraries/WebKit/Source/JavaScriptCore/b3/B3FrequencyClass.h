/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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

#if ENABLE(B3_JIT)

namespace JSC { namespace B3 {

enum class FrequencyClass : uint8_t {
    // We don't have any hypothesis about the frequency of this control flow construct. This is
    // the common case. We can still use basic block frequency in this case.
    Normal,

    // We expect that this control flow construct will be reached super rarely. It's valid to
    // perform optimizations that punish Rare code. Note that there will be situations where you
    // have to somehow construct a new frequency class from a merging of multiple classes. When
    // this happens, never choose Rare; always go with Normal. This is necessary because we
    // really do punish Rare code very badly.
    Rare
};

inline FrequencyClass maxFrequency(FrequencyClass a, FrequencyClass b)
{
    if (a == FrequencyClass::Normal)
        return FrequencyClass::Normal;
    return b;
}

} } // namespace JSC::B3

namespace WTF {

class PrintStream;

void printInternal(PrintStream&, JSC::B3::FrequencyClass);

} // namespace WTF

#endif // ENABLE(B3_JIT)
