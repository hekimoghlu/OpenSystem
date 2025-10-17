/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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

#include "Subspace.h"
#include "SubspaceAccess.h"

namespace JSC {

// Unlike other Subspaces, PreciseSubspace doesn't support LocalAllocators as it's meant for large, rarely allocated objects
// that wouldn't be profitable to allocate directly in JIT code.
class PreciseSubspace : public Subspace {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PreciseSubspace, JS_EXPORT_PRIVATE);
public:
    JS_EXPORT_PRIVATE PreciseSubspace(CString name, Heap&, const HeapCellType&, AlignedMemoryAllocator*);
    JS_EXPORT_PRIVATE ~PreciseSubspace() override;

    void* tryAllocate(size_t cellSize);
    void* allocate(VM&, size_t, GCDeferralContext*, AllocationFailureMode);

private:
    void didResizeBits(unsigned newSize) override;
    void didRemoveBlock(unsigned blockIndex) override;
    void didBeginSweepingToFreeList(MarkedBlock::Handle*) override;
};

namespace GCClient {
// This doesn't do anything interesting right now but we keep the GCClient namespace for consistency/templates.
using PreciseSubspace = JSC::PreciseSubspace;
}

} // namespace JSC

