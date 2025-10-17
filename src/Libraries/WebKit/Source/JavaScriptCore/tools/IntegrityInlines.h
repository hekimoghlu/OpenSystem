/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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

#include "Integrity.h"
#include "JSCJSValue.h"
#include "StructureID.h"
#include "VM.h"
#include "VMInspector.h"
#include <wtf/Atomics.h>
#include <wtf/Gigacage.h>

namespace JSC {
namespace Integrity {

ALWAYS_INLINE bool Random::shouldAudit(VM& vm)
{
    // If auditing is enabled, then the top bit of m_triggerBits is always set
    // to 1 on reload. When this top bit reaches the bottom, it does not
    // indicate that we should trigger an audit but rather that we've shifted
    // out all the available trigger bits and hence, need to reload. Instead,
    // reloadAndCheckShouldAuditSlow() will return whether we actually need to
    // trigger an audit this turn.
    //
    // This function can be called concurrently from different threads and can
    // be racy. For that reason, we intentionally do not write back to
    // m_triggerBits if newTriggerBits is null. This ensures that if
    // Options::randomIntegrityAuditRate() is non-zero, then m_triggerBits will
    // always have at least 1 bit to trigger a reload.

    uint64_t newTriggerBits = m_triggerBits;
    bool shouldAudit = newTriggerBits & 1;
    newTriggerBits = newTriggerBits >> 1;
    if (LIKELY(!shouldAudit)) {
        m_triggerBits = newTriggerBits;
        return false;
    }

    if (!newTriggerBits)
        return reloadAndCheckShouldAuditSlow(vm);

    m_triggerBits = newTriggerBits;
    return true;
}

template<AuditLevel auditLevel>
ALWAYS_INLINE void auditCell(VM& vm, JSValue value)
{
    if constexpr (auditLevel == AuditLevel::None)
        return;

    if (value.isCell())
        auditCell<auditLevel>(vm, value.asCell());
}

ALWAYS_INLINE void auditCellMinimally(VM& vm, JSCell* cell)
{
    if (UNLIKELY(Gigacage::contains(cell)))
        auditCellMinimallySlow(vm, cell);
}

ALWAYS_INLINE void auditCellRandomly(VM& vm, JSCell* cell)
{
    if (UNLIKELY(vm.integrityRandom().shouldAudit(vm)))
        auditCellFully(vm, cell);
}

ALWAYS_INLINE void auditCellFully(VM& vm, JSCell* cell)
{
#if USE(JSVALUE64)
    doAudit(vm, cell);
#else
    auditCellMinimally(vm, cell);
#endif
}

ALWAYS_INLINE void auditStructureID(StructureID structureID)
{
    UNUSED_PARAM(structureID);
#if CPU(ADDRESS64) && !ENABLE(STRUCTURE_ID_WITH_SHIFT)
    ASSERT(static_cast<uintptr_t>(structureID.bits()) <= structureHeapAddressSize + StructureID::nukedStructureIDBit);
#endif
#if ENABLE(EXTRA_INTEGRITY_CHECKS) || ASSERT_ENABLED
    Structure* structure = structureID.tryDecode();
    IA_ASSERT(structure, "structureID.bits 0x%x", structureID.bits());
    // structure should be pointing to readable memory. Force a read.
    WTF::opaque(*std::bit_cast<uintptr_t*>(structure));
#endif
}

#if USE(JSVALUE64)

JS_EXPORT_PRIVATE VM* doAuditSlow(VM*);

ALWAYS_INLINE VM* doAudit(VM* vm)
{
    if (UNLIKELY(!VMInspector::isValidVM(vm)))
        return doAuditSlow(vm);
    return vm;
}

#endif // USE(JSVALUE64)

} // namespace Integrity
} // namespace JSC
