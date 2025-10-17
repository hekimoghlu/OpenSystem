/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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

#include <wtf/ForbidHeapAllocation.h>

namespace JSC {

class VM;

// The only reason we implement DisallowVMEntry as specialization of a template
// is so that we can work around having to #include VM.h, which can hurt build
// time. This defers the cost of #include'ing VM.h to only the clients that
// need it.

template<typename VMType = VM>
class DisallowVMEntryImpl {
    WTF_FORBID_HEAP_ALLOCATION;
public:
    DisallowVMEntryImpl(VMType& vm)
        : m_vm(&vm)
    {
        m_vm->disallowVMEntryCount++;
    }

    DisallowVMEntryImpl(const DisallowVMEntryImpl& other)
        : m_vm(other.m_vm)
    {
        m_vm->disallowVMEntryCount++;
    }

    ~DisallowVMEntryImpl()
    {
        RELEASE_ASSERT(m_vm->disallowVMEntryCount);
        m_vm->disallowVMEntryCount--;
        m_vm = nullptr;
    }

    DisallowVMEntryImpl& operator=(const DisallowVMEntryImpl& other)
    {
        RELEASE_ASSERT(m_vm && m_vm == other.m_vm);
        RELEASE_ASSERT(m_vm->disallowVMEntryCount);
        // Conceptually, we need to decrement the disallowVMEntryCount of the
        // old m_vm, and increment the disallowVMEntryCount of the new m_vm.
        // But since the old and the new m_vm should always be the same, the
        // decrementing and incrementing cancels out, and there's nothing more
        // to do here.
        return *this;
    }

private:
    VMType* m_vm;
};

using DisallowVMEntry = DisallowVMEntryImpl<VM>;

} // namespace JSC
