/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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

#include "IsoSubspace.h"
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>

namespace JSC {

class Heap;
class VM;

// This is an appropriate way to stash IsoSubspaces for rarely-used classes or classes that are mostly
// sure to be main-thread-only. But if a class typically gets instantiated from multiple threads at
// once, then this is not great, because concurrent allocations will probably contend on this thing's
// lock.
//
// Usage: A IsoSubspacePerVM instance is instantiated using NeverDestroyed i.e. each IsoSubspacePerVM
// instance is immortal for the life of the process. There is one IsoSubspace type per IsoSubspacePerVM.
// IsoSubspacePerVM itself is a factory for those IsoSubspaces per VM and per Heap. IsoSubspacePerVM
// also serves as the manager of the IsoSubspaces. Client VMs are responsible for calling
// releaseClientIsoSubspace() to release the IsoSubspace when the VM shuts down. Similarly, if the
// Heap is not an immortal singleton heap (when global GC is enabled), the per Heap IsoSubspace also
// needs to be released when the Heap is destructed.
class IsoSubspacePerVM final {
public:
    struct SubspaceParameters {
        SubspaceParameters() { }
        
        SubspaceParameters(CString name, const HeapCellType& heapCellType, size_t size)
            : name(WTFMove(name))
            , heapCellType(&heapCellType)
            , size(size)
        {
        }
        
        CString name;
        const HeapCellType* heapCellType { nullptr };
        size_t size { 0 };
    };
    
    JS_EXPORT_PRIVATE IsoSubspacePerVM(Function<SubspaceParameters(Heap&)>);
    JS_EXPORT_PRIVATE ~IsoSubspacePerVM();
    
    JS_EXPORT_PRIVATE GCClient::IsoSubspace& clientIsoSubspaceforVM(VM&);

    // FIXME: GlobalGC: this is only needed until we have a immortal singleton heap with GlobalGC.
    void releaseIsoSubspace(Heap&);

    void releaseClientIsoSubspace(VM&);

private:
    IsoSubspace& isoSubspaceforHeap(Locker<Lock>&, Heap&);

    Lock m_lock;

    UncheckedKeyHashMap<Heap*, IsoSubspace*> m_subspacePerHeap;
    UncheckedKeyHashMap<VM*, GCClient::IsoSubspace*> m_clientSubspacePerVM WTF_GUARDED_BY_LOCK(m_lock);
    Function<SubspaceParameters(Heap&)> m_subspaceParameters;
};

#define ISO_SUBSPACE_PARAMETERS(heapCellType, type) ::JSC::IsoSubspacePerVM::SubspaceParameters("IsoSpace " #type, (heapCellType), sizeof(type))

} // namespace JSC

