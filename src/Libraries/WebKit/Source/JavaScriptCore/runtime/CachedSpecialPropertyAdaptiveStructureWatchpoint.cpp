/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
#include "CachedSpecialPropertyAdaptiveStructureWatchpoint.h"

#include "JSCellInlines.h"
#include "StructureRareData.h"

namespace JSC {

CachedSpecialPropertyAdaptiveStructureWatchpoint::CachedSpecialPropertyAdaptiveStructureWatchpoint(const ObjectPropertyCondition& key, StructureRareData* structureRareData)
    : Watchpoint(Watchpoint::Type::CachedSpecialPropertyAdaptiveStructure)
    , m_structureRareData(structureRareData)
    , m_key(key)
{
    RELEASE_ASSERT(key.watchingRequiresStructureTransitionWatchpoint());
    RELEASE_ASSERT(!key.watchingRequiresReplacementWatchpoint());
}

void CachedSpecialPropertyAdaptiveStructureWatchpoint::install(VM&)
{
    RELEASE_ASSERT(m_key.isWatchable(PropertyCondition::MakeNoChanges));

    m_key.object()->structure()->addTransitionWatchpoint(this);
}

void CachedSpecialPropertyAdaptiveStructureWatchpoint::fireInternal(VM& vm, const FireDetail&)
{
    if (m_structureRareData->isPendingDestruction())
        return;

    if (m_key.isWatchable(PropertyCondition::EnsureWatchability)) {
        install(vm);
        return;
    }

    CachedSpecialPropertyKey key = CachedSpecialPropertyKey::ToStringTag;
    if (m_key.uid() == vm.propertyNames->toStringTagSymbol.impl())
        key = CachedSpecialPropertyKey::ToStringTag;
    else if (m_key.uid() == vm.propertyNames->toString.impl())
        key = CachedSpecialPropertyKey::ToString;
    else if (m_key.uid() == vm.propertyNames->valueOf.impl())
        key = CachedSpecialPropertyKey::ValueOf;
    else if (m_key.uid() == vm.propertyNames->toJSON.impl())
        key = CachedSpecialPropertyKey::ToJSON;
    else {
        ASSERT(m_key.uid() == vm.propertyNames->toPrimitiveSymbol.impl());
        key = CachedSpecialPropertyKey::ToPrimitive;
    }

    m_structureRareData->clearCachedSpecialProperty(key);
}

} // namespace JSC
