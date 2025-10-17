/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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
#include "SimpleMarkingConstraint.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SimpleMarkingConstraint);

SimpleMarkingConstraint::SimpleMarkingConstraint(
    CString abbreviatedName, CString name,
    MarkingConstraintExecutorPair&& executors,
    ConstraintVolatility volatility, ConstraintConcurrency concurrency,
    ConstraintParallelism parallelism)
    : MarkingConstraint(WTFMove(abbreviatedName), WTFMove(name), volatility, concurrency, parallelism)
    , m_executors(WTFMove(executors))
{
}

SimpleMarkingConstraint::~SimpleMarkingConstraint() = default;

template<typename Visitor>
void SimpleMarkingConstraint::executeImplImpl(Visitor& visitor)
{
    m_executors.execute(visitor);
}

void SimpleMarkingConstraint::executeImpl(AbstractSlotVisitor& visitor) { executeImplImpl(visitor); }
void SimpleMarkingConstraint::executeImpl(SlotVisitor& visitor) { executeImplImpl(visitor); }

} // namespace JSC

