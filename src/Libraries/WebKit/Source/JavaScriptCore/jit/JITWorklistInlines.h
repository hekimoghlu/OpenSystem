/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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

#include "JITWorklist.h"

namespace JSC {

#if ENABLE(JIT)

template<typename Visitor>
void JITWorklist::iterateCodeBlocksForGC(Visitor& visitor, VM& vm, const Function<void(CodeBlock*)>& func)
{
    if (!vm.numberOfActiveJITPlans())
        return;

    Locker locker { *m_lock };
    for (auto& entry : m_plans) {
        if (entry.value->vm() != &vm)
            continue;
        entry.value->iterateCodeBlocksForGC(visitor, func);
    }
}

#endif // ENABLE(JIT)

} // namespace JSC
