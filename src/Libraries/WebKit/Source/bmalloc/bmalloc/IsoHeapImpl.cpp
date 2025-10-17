/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
#include "BPlatform.h"
#include "IsoHeapImpl.h"

#if !BUSE(TZONE)

#include "AllIsoHeaps.h"
#include "PerProcess.h"
#include <climits>

#if !BUSE(LIBPAS)

namespace bmalloc {

IsoHeapImplBase::IsoHeapImplBase(Mutex& lock)
    : lock(lock)
{
}

IsoHeapImplBase::~IsoHeapImplBase()
{
}

void IsoHeapImplBase::addToAllIsoHeaps()
{
    AllIsoHeaps::get()->add(this);
}

void IsoHeapImplBase::scavengeNow()
{
    Vector<DeferredDecommit> deferredDecommits;
    scavenge(deferredDecommits);
    finishScavenging(deferredDecommits);
}

void IsoHeapImplBase::finishScavenging(Vector<DeferredDecommit>& deferredDecommits)
{
    std::sort(
        deferredDecommits.begin(), deferredDecommits.end(),
        [&] (const DeferredDecommit& a, const DeferredDecommit& b) -> bool {
            return a.page < b.page;
        });
    unsigned runStartIndex = UINT_MAX;
    char* run = nullptr;
    size_t size = 0;
    auto resetRun = [&] (unsigned endIndex) {
        if (!run) {
            RELEASE_BASSERT(!size);
            RELEASE_BASSERT(runStartIndex == UINT_MAX);
            return;
        }
        RELEASE_BASSERT(size);
        RELEASE_BASSERT(runStartIndex != UINT_MAX);
        vmDeallocatePhysicalPages(run, size);
        for (unsigned i = runStartIndex; i < endIndex; ++i) {
            const DeferredDecommit& value = deferredDecommits[i];
            value.directory->didDecommit(value.pageIndex);
        }
        run = nullptr;
        size = 0;
        runStartIndex = UINT_MAX;
    };
    for (unsigned i = 0; i < deferredDecommits.size(); ++i) {
        const DeferredDecommit& value = deferredDecommits[i];
        char* page = reinterpret_cast<char*>(value.page);
        RELEASE_BASSERT(page >= run + size);
        if (page != run + size) {
            resetRun(i);
            runStartIndex = i;
            run = page;
        }
        size += IsoPageBase::pageSize;
    }
    resetRun(deferredDecommits.size());
}

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
