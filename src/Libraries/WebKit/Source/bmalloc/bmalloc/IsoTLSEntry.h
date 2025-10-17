/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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

#include "BAssert.h"
#include "BMalloced.h"
#include "IsoTLSLayout.h"
#include <climits>

#if !BUSE(LIBPAS)

namespace bmalloc {

class IsoTLS;

template<typename Entry>
class IsoTLSEntryHolder {
    MAKE_BMALLOCED;
    IsoTLSEntryHolder(const IsoTLSEntryHolder&) = delete;
    IsoTLSEntryHolder& operator=(const IsoTLSEntryHolder&) = delete;
public:
    template<typename... Args>
    IsoTLSEntryHolder(Args&&... args)
        : m_entry(std::forward<Args>(args)...)
    {
        IsoTLSLayout::get()->add(&m_entry);
        RELEASE_BASSERT(m_entry.offset() != UINT_MAX);
    }

    inline const Entry& operator*() const { m_entry; }
    inline Entry& operator*() { m_entry; }
    inline const Entry* operator->() const { return &m_entry; }
    inline Entry* operator->() { return &m_entry; }

private:
    Entry m_entry;
};

class BEXPORT IsoTLSEntry {
    MAKE_BMALLOCED;
    IsoTLSEntry(const IsoTLSEntry&) = delete;
    IsoTLSEntry& operator=(const IsoTLSEntry&) = delete;
public:
    virtual ~IsoTLSEntry();
    
    size_t offset() const { return m_offset; }
    size_t alignment() const { return sizeof(void*); }
    size_t size() const { return m_size; }
    size_t extent() const { return m_offset + m_size; }
    
    virtual void construct(void* entry) = 0;
    virtual void move(void* src, void* dst) = 0;
    virtual void destruct(void* entry) = 0;
    virtual void scavenge(void* entry) = 0;
    
    template<typename Func>
    void walkUpToInclusive(IsoTLSEntry*, const Func&);

protected:
    IsoTLSEntry(size_t size);
    
private:
    friend class IsoTLS;
    friend class IsoTLSLayout;

    IsoTLSEntry* m_next { nullptr };
    
    unsigned m_offset { UINT_MAX }; // Computed in constructor.
    unsigned m_size;
};

template<typename EntryType>
class DefaultIsoTLSEntry : public IsoTLSEntry {
public:
    ~DefaultIsoTLSEntry() = default;
    
protected:
    DefaultIsoTLSEntry();

    // This clones src onto dst and then destructs src. Therefore, entry destructors cannot do
    // scavenging.
    void move(void* src, void* dst) override;
    
    // Likewise, this is separate from scavenging. When the TLS is shutting down, we will be asked to
    // scavenge and then we will be asked to destruct.
    void destruct(void* entry) override;
};

} // namespace bmalloc

#endif
