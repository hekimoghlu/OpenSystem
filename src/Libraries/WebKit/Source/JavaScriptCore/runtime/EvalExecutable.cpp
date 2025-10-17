/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
#include "EvalExecutable.h"

#include "JSCJSValueInlines.h"

namespace JSC {

const ClassInfo EvalExecutable::s_info = { "EvalExecutable"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(EvalExecutable) };

EvalExecutable::EvalExecutable(JSGlobalObject* globalObject, const SourceCode& source, LexicallyScopedFeatures lexicallyScopedFeatures, DerivedContextType derivedContextType, bool isArrowFunctionContext, bool isInsideOrdinaryFunction, EvalContextType evalContextType, NeedsClassFieldInitializer needsClassFieldInitializer, PrivateBrandRequirement privateBrandRequirement)
    : Base(globalObject->vm().evalExecutableStructure.get(), globalObject->vm(), source, lexicallyScopedFeatures, derivedContextType, isArrowFunctionContext, isInsideOrdinaryFunction, evalContextType, NoIntrinsic)
    , m_needsClassFieldInitializer(static_cast<unsigned>(needsClassFieldInitializer))
    , m_privateBrandRequirement(static_cast<unsigned>(privateBrandRequirement))
{
    ASSERT(source.provider()->sourceType() == SourceProviderSourceType::Program);
}

void EvalExecutable::destroy(JSCell* cell)
{
    static_cast<EvalExecutable*>(cell)->EvalExecutable::~EvalExecutable();
}

auto EvalExecutable::ensureTemplateObjectMap(VM&) -> TemplateObjectMap&
{
    return ensureTemplateObjectMapImpl(m_templateObjectMap);
}

template<typename Visitor>
void EvalExecutable::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    EvalExecutable* thisObject = jsCast<EvalExecutable*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    if (TemplateObjectMap* map = thisObject->m_templateObjectMap.get()) {
        Locker locker { thisObject->cellLock() };
        for (auto& entry : *map)
            visitor.append(entry.value);
    }
}

DEFINE_VISIT_CHILDREN(EvalExecutable);

} // namespace JSC
