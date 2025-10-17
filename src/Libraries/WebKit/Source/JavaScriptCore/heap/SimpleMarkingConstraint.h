/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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

#include "MarkingConstraint.h"
#include "MarkingConstraintExecutorPair.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

// This allows for an informal way to define constraints. Just pass a lambda to the constructor. The only
// downside is that this makes it hard for constraints to override any functions in MarkingConstraint
// other than executeImpl. In those cases, just subclass MarkingConstraint.
class SimpleMarkingConstraint final : public MarkingConstraint {
    WTF_MAKE_TZONE_ALLOCATED(SimpleMarkingConstraint);
public:
    JS_EXPORT_PRIVATE SimpleMarkingConstraint(
        CString abbreviatedName, CString name,
        MarkingConstraintExecutorPair&&,
        ConstraintVolatility,
        ConstraintConcurrency = ConstraintConcurrency::Concurrent,
        ConstraintParallelism = ConstraintParallelism::Sequential);
    
    SimpleMarkingConstraint(
        CString abbreviatedName, CString name,
        MarkingConstraintExecutorPair&& executors,
        ConstraintVolatility volatility,
        ConstraintParallelism parallelism)
        : SimpleMarkingConstraint(abbreviatedName, name, WTFMove(executors), volatility, ConstraintConcurrency::Concurrent, parallelism)
    {
    }
    
    JS_EXPORT_PRIVATE ~SimpleMarkingConstraint() final;
    
private:
    template<typename Visitor> ALWAYS_INLINE void executeImplImpl(Visitor&);
    void executeImpl(AbstractSlotVisitor&) final;
    void executeImpl(SlotVisitor&) final;

    MarkingConstraintExecutorPair m_executors;
};

} // namespace JSC

