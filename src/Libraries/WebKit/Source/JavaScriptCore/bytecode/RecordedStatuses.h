/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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

#include "CallLinkStatus.h"
#include "CheckPrivateBrandStatus.h"
#include "DeleteByStatus.h"
#include "GetByStatus.h"
#include "InByStatus.h"
#include "PutByStatus.h"
#include "SetPrivateBrandStatus.h"

namespace JSC {

struct RecordedStatuses {
    WTF_MAKE_STRUCT_FAST_ALLOCATED(RecordedStatuses);

    RecordedStatuses() = default;

    CallLinkStatus* addCallLinkStatus(const CodeOrigin&, const CallLinkStatus&);
    GetByStatus* addGetByStatus(const CodeOrigin&, const GetByStatus&);
    PutByStatus* addPutByStatus(const CodeOrigin&, const PutByStatus&);
    InByStatus* addInByStatus(const CodeOrigin&, const InByStatus&);
    DeleteByStatus* addDeleteByStatus(const CodeOrigin&, const DeleteByStatus&);
    CheckPrivateBrandStatus* addCheckPrivateBrandStatus(const CodeOrigin&, const CheckPrivateBrandStatus&);
    SetPrivateBrandStatus* addSetPrivateBrandStatus(const CodeOrigin&, const SetPrivateBrandStatus&);
    
    DECLARE_VISIT_AGGREGATE;
    template<typename Visitor> void markIfCheap(Visitor&);
    
    void finalizeWithoutDeleting(VM&);
    void finalize(VM&);
    
    void shrinkToFit();
    
    template<typename Func>
    void forEachVector(const Func& func)
    {
        func(calls);
        func(gets);
        func(puts);
        func(ins);
        func(deletes);
        func(checkPrivateBrands);
        func(setPrivateBrands);
    }
    
    Vector<std::pair<CodeOrigin, std::unique_ptr<CallLinkStatus>>> calls;
    Vector<std::pair<CodeOrigin, std::unique_ptr<GetByStatus>>> gets;
    Vector<std::pair<CodeOrigin, std::unique_ptr<PutByStatus>>> puts;
    Vector<std::pair<CodeOrigin, std::unique_ptr<InByStatus>>> ins;
    Vector<std::pair<CodeOrigin, std::unique_ptr<DeleteByStatus>>> deletes;
    Vector<std::pair<CodeOrigin, std::unique_ptr<CheckPrivateBrandStatus>>> checkPrivateBrands;
    Vector<std::pair<CodeOrigin, std::unique_ptr<SetPrivateBrandStatus>>> setPrivateBrands;
};

} // namespace JSC

