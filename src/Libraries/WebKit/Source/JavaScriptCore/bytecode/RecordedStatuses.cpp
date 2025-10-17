/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#include "RecordedStatuses.h"

namespace JSC {

CallLinkStatus* RecordedStatuses::addCallLinkStatus(const CodeOrigin& codeOrigin, const CallLinkStatus& status)
{
    auto statusPtr = makeUnique<CallLinkStatus>(status);
    CallLinkStatus* result = statusPtr.get();
    calls.append(std::make_pair(codeOrigin, WTFMove(statusPtr)));
    return result;
}

GetByStatus* RecordedStatuses::addGetByStatus(const CodeOrigin& codeOrigin, const GetByStatus& status)
{
    auto statusPtr = makeUnique<GetByStatus>(status);
    GetByStatus* result = statusPtr.get();
    gets.append(std::make_pair(codeOrigin, WTFMove(statusPtr)));
    return result;
}
    
PutByStatus* RecordedStatuses::addPutByStatus(const CodeOrigin& codeOrigin, const PutByStatus& status)
{
    auto statusPtr = makeUnique<PutByStatus>(status);
    PutByStatus* result = statusPtr.get();
    puts.append(std::make_pair(codeOrigin, WTFMove(statusPtr)));
    return result;
}

InByStatus* RecordedStatuses::addInByStatus(const CodeOrigin& codeOrigin, const InByStatus& status)
{
    auto statusPtr = makeUnique<InByStatus>(status);
    InByStatus* result = statusPtr.get();
    ins.append(std::make_pair(codeOrigin, WTFMove(statusPtr)));
    return result;
}

DeleteByStatus* RecordedStatuses::addDeleteByStatus(const CodeOrigin& codeOrigin, const DeleteByStatus& status)
{
    auto statusPtr = makeUnique<DeleteByStatus>(status);
    DeleteByStatus* result = statusPtr.get();
    deletes.append(std::make_pair(codeOrigin, WTFMove(statusPtr)));
    return result;
}

CheckPrivateBrandStatus* RecordedStatuses::addCheckPrivateBrandStatus(const CodeOrigin& codeOrigin, const CheckPrivateBrandStatus& status)
{
    auto statusPtr = makeUnique<CheckPrivateBrandStatus>(status);
    CheckPrivateBrandStatus* result = statusPtr.get();
    checkPrivateBrands.append(std::make_pair(codeOrigin, WTFMove(statusPtr)));
    return result;
}

SetPrivateBrandStatus* RecordedStatuses::addSetPrivateBrandStatus(const CodeOrigin& codeOrigin, const SetPrivateBrandStatus& status)
{
    auto statusPtr = makeUnique<SetPrivateBrandStatus>(status);
    SetPrivateBrandStatus* result = statusPtr.get();
    setPrivateBrands.append(std::make_pair(codeOrigin, WTFMove(statusPtr)));
    return result;
}

template<typename Visitor>
void RecordedStatuses::visitAggregateImpl(Visitor& visitor)
{
    for (auto& pair : gets)
        pair.second->visitAggregate(visitor);
    for (auto& pair : puts)
        pair.second->visitAggregate(visitor);
    for (auto& pair : ins)
        pair.second->visitAggregate(visitor);
    for (auto& pair : deletes)
        pair.second->visitAggregate(visitor);
    for (auto& pair : checkPrivateBrands)
        pair.second->visitAggregate(visitor);
    for (auto& pair : setPrivateBrands)
        pair.second->visitAggregate(visitor);
}

DEFINE_VISIT_AGGREGATE(RecordedStatuses);

template<typename Visitor>
void RecordedStatuses::markIfCheap(Visitor& visitor)
{
    for (auto& pair : gets)
        pair.second->markIfCheap(visitor);
    for (auto& pair : puts)
        pair.second->markIfCheap(visitor);
    for (auto& pair : ins)
        pair.second->markIfCheap(visitor);
    for (auto& pair : deletes)
        pair.second->markIfCheap(visitor);
    for (auto& pair : checkPrivateBrands)
        pair.second->markIfCheap(visitor);
    for (auto& pair : setPrivateBrands)
        pair.second->markIfCheap(visitor);
}

template void RecordedStatuses::markIfCheap(AbstractSlotVisitor&);
template void RecordedStatuses::markIfCheap(SlotVisitor&);

void RecordedStatuses::finalizeWithoutDeleting(VM& vm)
{
    // This variant of finalize gets called from within graph safepoints -- so there may be DFG IR in
    // some compiler thread that points to the statuses. That thread is stopped at a safepoint so
    // it's OK to edit its data structure, but it's not OK to delete them. Hence we don't remove
    // anything from the vector or delete the unique_ptrs.
    
    auto finalize = [&] (auto& vector) {
        for (auto& pair : vector) {
            if (!pair.second->finalize(vm))
                *pair.second = { };
        }
    };
    forEachVector(finalize);
}

void RecordedStatuses::finalize(VM& vm)
{
    auto finalize = [&] (auto& vector) {
        vector.removeAllMatching(
            [&] (auto& pair) -> bool {
                return !*pair.second || !pair.second->finalize(vm);
            });
        vector.shrinkToFit();
    };
    forEachVector(finalize);
}

void RecordedStatuses::shrinkToFit()
{
    forEachVector([] (auto& vector) { vector.shrinkToFit(); });
}

} // namespace JSC

