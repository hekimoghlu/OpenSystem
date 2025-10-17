/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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

#if !BUSE(TZONE)

#include "BExport.h"
#include <cstddef>
#include <cstdint>

#if !BUSE(LIBPAS)

namespace bmalloc {

class VariadicBumpAllocator;

struct FreeCell {
    static uintptr_t scramble(FreeCell* cell, uintptr_t secret)
    {
        return reinterpret_cast<uintptr_t>(cell) ^ secret;
    }
    
    static FreeCell* descramble(uintptr_t cell, uintptr_t secret)
    {
        return reinterpret_cast<FreeCell*>(cell ^ secret);
    }
    
    void setNext(FreeCell* next, uintptr_t secret)
    {
        scrambledNext = scramble(next, secret);
    }
    
    FreeCell* next(uintptr_t secret) const
    {
        return descramble(scrambledNext, secret);
    }
    
    uintptr_t scrambledNext;
};

class FreeList {
public:
    friend class VariadicBumpAllocator;

    BEXPORT FreeList();
    BEXPORT ~FreeList();
    
    BEXPORT void clear();
    
    BEXPORT void initializeList(FreeCell* head, uintptr_t secret, unsigned bytes);
    BEXPORT void initializeBump(char* payloadEnd, unsigned remaining);
    
    bool allocationWillFail() const { return !head() && !m_remaining; }
    bool allocationWillSucceed() const { return !allocationWillFail(); }
    
    template<typename Config, typename Func>
    void* allocate(const Func& slowPath);
    
    bool contains(void*) const;
    
    template<typename Config, typename Func>
    void forEach(const Func&) const;
    
    unsigned originalSize() const { return m_originalSize; }

private:
    FreeCell* head() const { return FreeCell::descramble(m_scrambledHead, m_secret); }
    
    uintptr_t m_scrambledHead { 0 };
    uintptr_t m_secret { 0 };
    char* m_payloadEnd { nullptr };
    unsigned m_remaining { 0 };
    unsigned m_originalSize { 0 };
};

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
