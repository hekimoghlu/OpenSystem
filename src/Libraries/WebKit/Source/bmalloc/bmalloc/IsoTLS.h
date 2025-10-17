/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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

#include "PerThread.h"
#include <cstddef>

#if !BUSE(LIBPAS)

namespace bmalloc {

class IsoTLSEntry;
template<typename Config> class IsoAllocator;
template<typename Config> class IsoDeallocator;

namespace api {
template<typename Type> struct IsoHeapBase;
}

class IsoTLS {
public:
    template<typename Type>
    static void* allocate(api::IsoHeapBase<Type>&, bool abortOnFailure);
    
    template<typename Type>
    static void deallocate(api::IsoHeapBase<Type>&, void* p);
    
    template<typename Type>
    static void ensureHeap(api::IsoHeapBase<Type>&);
    
    BEXPORT static void scavenge();
    
    template<typename Type>
    static void scavenge(api::IsoHeapBase<Type>&);

private:
    IsoTLS();
    
    template<typename Config, typename Type>
    static void* allocateImpl(api::IsoHeapBase<Type>&, bool abortOnFailure);
    
    template<typename Config, typename Type>
    void* allocateFast(api::IsoHeapBase<Type>&, unsigned offset, bool abortOnFailure);
    
    template<typename Config, typename Type>
    static void* allocateSlow(api::IsoHeapBase<Type>&, bool abortOnFailure);
    
    template<typename Config, typename Type>
    static void deallocateImpl(api::IsoHeapBase<Type>&, void* p);
    
    template<typename Config, typename Type>
    void deallocateFast(api::IsoHeapBase<Type>&, unsigned offset, void* p);
    
    template<typename Config, typename Type>
    static void deallocateSlow(api::IsoHeapBase<Type>&, void* p);
    
    static IsoTLS* get();
    static void set(IsoTLS*);
    
    template<typename Type>
    static IsoTLS* ensureHeapAndEntries(api::IsoHeapBase<Type>&);
    
    BEXPORT static IsoTLS* ensureEntries(unsigned offset);

    static void destructor(void* arg); // FIXME implement this
    
    static size_t sizeForCapacity(unsigned capacity);
    static unsigned capacityForSize(size_t size);
    
    size_t size();
    
    template<typename Func>
    void forEachEntry(const Func&);
    
    IsoTLSEntry* m_lastEntry { nullptr };
    unsigned m_extent { 0 };
    unsigned m_capacity { 0 };
    char m_data[1];

#if HAVE_PTHREAD_MACHDEP_H
    static constexpr pthread_key_t tlsKey = __PTK_FRAMEWORK_JAVASCRIPTCORE_KEY1;
#else
    BEXPORT static bool s_didInitialize;
    BEXPORT static pthread_key_t s_tlsKey;
#endif
};

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
