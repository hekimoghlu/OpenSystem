/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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

#include "Bits.h"
#include "EligibilityResult.h"
#include "IsoPage.h"
#include "Vector.h"

namespace bmalloc {

template<typename Config> class IsoHeapImpl;

class IsoDirectoryBaseBase {
public:
    IsoDirectoryBaseBase() { }
    virtual ~IsoDirectoryBaseBase() { }

    virtual void didDecommit(unsigned index) = 0;
};

template<typename Config>
class IsoDirectoryBase : public IsoDirectoryBaseBase {
public:
    IsoDirectoryBase(IsoHeapImpl<Config>&);
    
    IsoHeapImpl<Config>& heap() { return m_heap; }
    
    virtual void didBecome(IsoPage<Config>*, IsoPageTrigger) = 0;
    
protected:
    IsoHeapImpl<Config>& m_heap;
};

template<typename Config, unsigned passedNumPages>
class IsoDirectory : public IsoDirectoryBase<Config> {
public:
    static constexpr unsigned numPages = passedNumPages;
    
    IsoDirectory(IsoHeapImpl<Config>&);
    
    // Find the first page that is eligible for allocation and return it. May return null if there is no
    // such thing. May allocate a new page if we have an uncommitted page.
    EligibilityResult<Config> takeFirstEligible();
    
    void didBecome(IsoPage<Config>*, IsoPageTrigger) override;
    
    // This gets called from a bulk decommit function in the Scavenger, so no locks are held. This function
    // needs to get the heap lock.
    void didDecommit(unsigned index) override;
    
    // Iterate over all empty and committed pages, and put them into the vector. This also records the
    // pages as being decommitted. It's the caller's job to do the actual decommitting.
    void scavenge(Vector<DeferredDecommit>&);
    void scavengeToHighWatermark(Vector<DeferredDecommit>&);

    template<typename Func>
    void forEachCommittedPage(const Func&);
    
private:
    void scavengePage(size_t, Vector<DeferredDecommit>&);

    // NOTE: I suppose that this could be two bitvectors. But from working on the GC, I found that the
    // number of bitvectors does not matter as much as whether or not they make intuitive sense.
    Bits<numPages> m_eligible;
    Bits<numPages> m_empty;
    Bits<numPages> m_committed;
    std::array<IsoPage<Config>*, numPages> m_pages;
    unsigned m_firstEligible { 0 };
    unsigned m_highWatermark { 0 };
};

} // namespace bmalloc

