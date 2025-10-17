/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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

#include "VM.h"
#include <wtf/ForbidHeapAllocation.h>
#include <wtf/Noncopyable.h>

namespace JSC {

template<VMTraps::DeferAction deferAction = VMTraps::DeferAction::DeferUntilEndOfScope>
class DeferTermination {
    WTF_MAKE_NONCOPYABLE(DeferTermination);
    WTF_FORBID_HEAP_ALLOCATION;
public:
    DeferTermination(VM& vm)
        : m_vm(vm)
    {
        m_vm.traps().deferTermination(deferAction);
    }
    
    ~DeferTermination()
    {
        m_vm.traps().undoDeferTermination(deferAction);
    }

private:
    VM& m_vm;
};

using DeferTerminationForAWhile = DeferTermination<VMTraps::DeferAction::DeferForAWhile>;

} // namespace JSC
