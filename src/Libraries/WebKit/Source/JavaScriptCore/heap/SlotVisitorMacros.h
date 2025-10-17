/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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

namespace JSC {

class AbstractSlotVisitor;
class JSCell;
class SlotVisitor;

#define UNUSED_MODIFIER_ARGUMENT /* unused */

#define DECLARE_UNUSED_STRUCT_TO_TERMINATE_VISIT_MACRO_EXPANDER_IMPL(counter) \
    struct UnusedSlotVisitorMacroStructSoWeCanHaveASemicolon##counter { }

#define DECLARE_UNUSED_STRUCT_TO_TERMINATE_VISIT_MACRO_EXPANDER(counter) \
    DECLARE_UNUSED_STRUCT_TO_TERMINATE_VISIT_MACRO_EXPANDER_IMPL(counter)

#define DECLARE_UNUSED_STRUCT_TO_TERMINATE_VISIT_MACRO \
    DECLARE_UNUSED_STRUCT_TO_TERMINATE_VISIT_MACRO_EXPANDER(__COUNTER__)

// Macros for visitAggregate().

#define DECLARE_VISIT_AGGREGATE_WITH_MODIFIER(postModifier) \
private: \
    template<typename Visitor> ALWAYS_INLINE void visitAggregateImpl(Visitor&) postModifier; \
public: \
    void visitAggregate(AbstractSlotVisitor&) postModifier; \
    void visitAggregate(SlotVisitor&) postModifier

#define DECLARE_VISIT_AGGREGATE \
    DECLARE_VISIT_AGGREGATE_WITH_MODIFIER(UNUSED_MODIFIER_ARGUMENT)

#define DEFINE_VISIT_AGGREGATE_WITH_MODIFIER(className, postModifier) \
    void className::visitAggregate(AbstractSlotVisitor& visitor) postModifier { visitAggregateImpl(visitor); } \
    void className::visitAggregate(SlotVisitor& visitor) postModifier { visitAggregateImpl(visitor); } \
    DECLARE_UNUSED_STRUCT_TO_TERMINATE_VISIT_MACRO

#define DEFINE_VISIT_AGGREGATE(className) \
    DEFINE_VISIT_AGGREGATE_WITH_MODIFIER(className, UNUSED_MODIFIER_ARGUMENT)

// Macros for static visitChildren().

#define DECLARE_VISIT_CHILDREN_WITH_MODIFIER(preModifier) \
private: \
    template<typename Visitor> ALWAYS_INLINE static void visitChildrenImpl(JSCell*, Visitor&); \
public: \
    preModifier static void visitChildren(JSCell*, AbstractSlotVisitor&); \
    preModifier static void visitChildren(JSCell*, SlotVisitor&)

#define DECLARE_VISIT_CHILDREN \
    DECLARE_VISIT_CHILDREN_WITH_MODIFIER(UNUSED_MODIFIER_ARGUMENT)

#define DEFINE_VISIT_CHILDREN_WITH_MODIFIER(preModifier, className) \
    preModifier void className::visitChildren(JSCell* cell, AbstractSlotVisitor& visitor) \
    { \
        AbstractSlotVisitor::ReferrerContext context(visitor, cell); \
        visitChildrenImpl(cell, visitor); \
    } \
    preModifier void className::visitChildren(JSCell* cell, SlotVisitor& visitor) { visitChildrenImpl(cell, visitor); } \
    DECLARE_UNUSED_STRUCT_TO_TERMINATE_VISIT_MACRO

#define DEFINE_VISIT_CHILDREN(className) \
    DEFINE_VISIT_CHILDREN_WITH_MODIFIER(UNUSED_MODIFIER_ARGUMENT, className)

// Macros for static visitOutputConstraints().

#define DECLARE_VISIT_OUTPUT_CONSTRAINTS_WITH_MODIFIER(preModifier) \
private: \
    template<typename Visitor> ALWAYS_INLINE static void visitOutputConstraintsImpl(JSCell*, Visitor&); \
public: \
    preModifier static void visitOutputConstraints(JSCell*, AbstractSlotVisitor&); \
    preModifier static void visitOutputConstraints(JSCell*, SlotVisitor&)

#define DECLARE_VISIT_OUTPUT_CONSTRAINTS \
    DECLARE_VISIT_OUTPUT_CONSTRAINTS_WITH_MODIFIER(UNUSED_MODIFIER_ARGUMENT)

#define DEFINE_VISIT_OUTPUT_CONSTRAINTS_WITH_MODIFIER(preModifier, className) \
    preModifier void className::visitOutputConstraints(JSCell* cell, AbstractSlotVisitor& visitor) \
    { \
        AbstractSlotVisitor::ReferrerContext context(visitor, cell); \
        visitOutputConstraintsImpl(cell, visitor); \
    } \
    preModifier void className::visitOutputConstraints(JSCell* cell, SlotVisitor& visitor) { visitOutputConstraintsImpl(cell, visitor); } \
    DECLARE_UNUSED_STRUCT_TO_TERMINATE_VISIT_MACRO

#define DEFINE_VISIT_OUTPUT_CONSTRAINTS(className) \
    DEFINE_VISIT_OUTPUT_CONSTRAINTS_WITH_MODIFIER(UNUSED_MODIFIER_ARGUMENT, className)

// Macros for visitAdditionalChildren().

#define DEFINE_VISIT_ADDITIONAL_CHILDREN(className) \
    template void className::visitAdditionalChildren(JSC::AbstractSlotVisitor&); \
    template void className::visitAdditionalChildren(JSC::SlotVisitor&)

} // namespace JSC

using JSC::AbstractSlotVisitor;
using JSC::JSCell;
using JSC::SlotVisitor;
