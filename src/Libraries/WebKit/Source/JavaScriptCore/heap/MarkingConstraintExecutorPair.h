/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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

#include <wtf/Function.h>

namespace JSC {

class AbstractSlotVisitor;
class SlotVisitor;

struct MarkingConstraintExecutorPair {
    MarkingConstraintExecutorPair(
        ::Function<void(AbstractSlotVisitor&)> abstractSlotVisitorFunc,
        ::Function<void(SlotVisitor&)> slotVisitorFunc)
        : abstractSlotVisitorFunc(WTFMove(abstractSlotVisitorFunc))
        , slotVisitorFunc(WTFMove(slotVisitorFunc))
    { }
    MarkingConstraintExecutorPair(MarkingConstraintExecutorPair&&) = default;

    void execute(AbstractSlotVisitor& visitor) { abstractSlotVisitorFunc(visitor); }
    void execute(SlotVisitor& visitor) { slotVisitorFunc(visitor); }

    ::Function<void(AbstractSlotVisitor&)> abstractSlotVisitorFunc;
    ::Function<void(SlotVisitor&)> slotVisitorFunc;
};

#define MAKE_MARKING_CONSTRAINT_EXECUTOR_PAIR(lambda) \
    MarkingConstraintExecutorPair((lambda), (lambda))

} // namespace JSC
