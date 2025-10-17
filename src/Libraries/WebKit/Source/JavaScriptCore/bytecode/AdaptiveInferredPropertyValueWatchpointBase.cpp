/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#include "AdaptiveInferredPropertyValueWatchpointBase.h"

#include "JSCInlines.h"
#include <wtf/TZoneMallocInlines.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AdaptiveInferredPropertyValueWatchpointBase);

AdaptiveInferredPropertyValueWatchpointBase::AdaptiveInferredPropertyValueWatchpointBase(const ObjectPropertyCondition& key)
    : m_key(key)
{
    RELEASE_ASSERT(key.kind() == PropertyCondition::Equivalence);
}

void AdaptiveInferredPropertyValueWatchpointBase::initialize(const ObjectPropertyCondition& key)
{
    m_key = key;
    RELEASE_ASSERT(key.kind() == PropertyCondition::Equivalence);
}

void AdaptiveInferredPropertyValueWatchpointBase::install(VM& vm)
{
    ASSERT(m_key.isWatchable(PropertyCondition::MakeNoChanges)); // This is really costly.

    Structure* structure = m_key.object()->structure();

    structure->addTransitionWatchpoint(&m_structureWatchpoint);

    PropertyOffset offset = structure->get(vm, m_key.uid());
    WatchpointSet* set = structure->propertyReplacementWatchpointSet(offset);
    set->add(&m_propertyWatchpoint);
}

void AdaptiveInferredPropertyValueWatchpointBase::fire(VM& vm, const FireDetail& detail)
{
    // One of the watchpoints fired, but the other one didn't. Make sure that neither of them are
    // in any set anymore. This simplifies things by allowing us to reinstall the watchpoints
    // wherever from scratch.
    if (m_structureWatchpoint.isOnList())
        m_structureWatchpoint.remove();
    if (m_propertyWatchpoint.isOnList())
        m_propertyWatchpoint.remove();

    if (!isValid())
        return;

    if (m_key.isWatchable(PropertyCondition::EnsureWatchability)) {
        install(vm);
        return;
    }

    handleFire(vm, detail);
}

bool AdaptiveInferredPropertyValueWatchpointBase::isValid() const
{
    return true;
}

void AdaptiveInferredPropertyValueWatchpointBase::StructureWatchpoint::fireInternal(VM& vm, const FireDetail& detail)
{
    ptrdiff_t myOffset = OBJECT_OFFSETOF(AdaptiveInferredPropertyValueWatchpointBase, m_structureWatchpoint);

    AdaptiveInferredPropertyValueWatchpointBase* parent = std::bit_cast<AdaptiveInferredPropertyValueWatchpointBase*>(std::bit_cast<char*>(this) - myOffset);

    parent->fire(vm, detail);
}

void AdaptiveInferredPropertyValueWatchpointBase::PropertyWatchpoint::fireInternal(VM& vm, const FireDetail& detail)
{
    ptrdiff_t myOffset = OBJECT_OFFSETOF(AdaptiveInferredPropertyValueWatchpointBase, m_propertyWatchpoint);

    AdaptiveInferredPropertyValueWatchpointBase* parent = std::bit_cast<AdaptiveInferredPropertyValueWatchpointBase*>(std::bit_cast<char*>(this) - myOffset);
    
    parent->fire(vm, detail);
}
    
} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
