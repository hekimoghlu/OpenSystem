/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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

#include "BMalloced.h"

namespace bmalloc {

class IsoTLS;
class IsoTLSLayout;

class BEXPORT IsoTLSEntry {
    MAKE_BMALLOCED;
public:
    IsoTLSEntry(size_t alignment, size_t size);
    virtual ~IsoTLSEntry();
    
    size_t offset() const { return m_offset; }
    size_t alignment() const { return m_alignment; }
    size_t size() const { return m_size; }
    size_t extent() const { return m_offset + m_size; }
    
    virtual void construct(void* entry) = 0;
    virtual void move(void* src, void* dst) = 0;
    virtual void destruct(void* entry) = 0;
    virtual void scavenge(void* entry) = 0;
    
    template<typename Func>
    void walkUpToInclusive(IsoTLSEntry*, const Func&);
    
private:
    friend class IsoTLS;
    friend class IsoTLSLayout;

    IsoTLSEntry* m_next { nullptr };
    
    size_t m_offset; // Computed in constructor.
    size_t m_alignment;
    size_t m_size;
};

template<typename EntryType>
class DefaultIsoTLSEntry : public IsoTLSEntry {
public:
    DefaultIsoTLSEntry();
    ~DefaultIsoTLSEntry();
    
protected:
    // This clones src onto dst and then destructs src. Therefore, entry destructors cannot do
    // scavenging.
    void move(void* src, void* dst) override;
    
    // Likewise, this is separate from scavenging. When the TLS is shutting down, we will be asked to
    // scavenge and then we will be asked to destruct.
    void destruct(void* entry) override;

    void scavenge(void* entry) override;
};

} // namespace bmalloc

