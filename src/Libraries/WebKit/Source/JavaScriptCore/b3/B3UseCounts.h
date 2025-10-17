/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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

#include "B3Value.h"
#include <wtf/IndexMap.h>

namespace JSC { namespace B3 {

class Procedure;

class UseCounts {
public:
    JS_EXPORT_PRIVATE UseCounts(Procedure&);
    JS_EXPORT_PRIVATE ~UseCounts();

    unsigned numUses(Value* value) const { return m_counts[value].numUses; }
    unsigned numUsingInstructions(Value* value) const { return m_counts[value].numUsingInstructions; }
    
private:
    struct Counts {
        unsigned numUses { 0 };
        unsigned numUsingInstructions { 0 };
    };
    
    IndexMap<Value*, Counts> m_counts;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
