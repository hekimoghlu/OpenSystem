/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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
#include "B3UseCounts.h"

#if ENABLE(B3_JIT)

#include "B3Procedure.h"
#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

UseCounts::UseCounts(Procedure& procedure)
    : m_counts(procedure.values().size())
{
    Vector<Value*, 64> children;
    for (Value* value : procedure.values()) {
        children.shrink(0);
        for (Value* child : value->children()) {
            m_counts[child].numUses++;
            children.append(child);
        }
        std::sort(children.begin(), children.end());
        Value* last = nullptr;
        for (Value* child : children) {
            if (child == last)
                continue;

            m_counts[child].numUsingInstructions++;
            last = child;
        }
    }
}

UseCounts::~UseCounts() = default;

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
