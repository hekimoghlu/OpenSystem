/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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

#include "FreeList.h"

#if !BUSE(LIBPAS)

namespace bmalloc {

template<typename Config> class IsoPage;
template<typename Config> class IsoHeapImpl;

enum class AllocationMode : uint8_t { Init, Fast, Shared };

template<typename Config>
class IsoAllocator {
public:
    IsoAllocator(IsoHeapImpl<Config>&);
    ~IsoAllocator();
    
    void* allocate(IsoHeapImpl<Config>&, bool abortOnFailure);
    void scavenge(IsoHeapImpl<Config>&);
    
private:
    void* allocateSlow(IsoHeapImpl<Config>&, bool abortOnFailure);
    
    FreeList m_freeList;
    IsoPage<Config>* m_currentPage { nullptr };
};

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
