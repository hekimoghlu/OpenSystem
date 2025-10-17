/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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

#include "BInline.h"
#include "IsoDeallocator.h"
#include "IsoPage.h"
#include "Mutex.h"
#include <mutex>

namespace bmalloc {

template<typename Config>
IsoDeallocator<Config>::IsoDeallocator(Mutex& lock)
    : m_lock(&lock)
{
}

template<typename Config>
IsoDeallocator<Config>::~IsoDeallocator()
{
}

template<typename Config>
void IsoDeallocator<Config>::deallocate(void* ptr)
{
    static constexpr bool verbose = false;
    if (verbose)
        fprintf(stderr, "%p: deallocating %p of size %u\n", &IsoPage<Config>::pageFor(ptr)->heap(), ptr, Config::objectSize);

    if (m_objectLog.size() == m_objectLog.capacity())
        scavenge();
    
    m_objectLog.push(ptr);
}

template<typename Config>
BNO_INLINE void IsoDeallocator<Config>::scavenge()
{
    std::lock_guard<Mutex> locker(*m_lock);
    
    for (void* ptr : m_objectLog)
        IsoPage<Config>::pageFor(ptr)->free(ptr);
    m_objectLog.clear();
}

} // namespace bmalloc

