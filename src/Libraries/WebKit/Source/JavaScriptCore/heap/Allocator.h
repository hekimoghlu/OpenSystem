/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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

#include "AllocationFailureMode.h"
#include <climits>

namespace JSC {

class GCDeferralContext;
class Heap;
class LocalAllocator;

// This abstracts how we refer to LocalAllocator so that we could eventually support thread-local
// caches.

class Allocator {
public:
    Allocator() { }
    
    explicit Allocator(LocalAllocator* localAllocator)
        : m_localAllocator(localAllocator)
    {
    }
    
    void* allocate(Heap&, size_t cellSize, GCDeferralContext*, AllocationFailureMode) const;
    
    unsigned cellSize() const;
    
    LocalAllocator* localAllocator() const { return m_localAllocator; }
    
    friend bool operator==(const Allocator&, const Allocator&) = default;
    explicit operator bool() const { return *this != Allocator(); }
    
private:
    LocalAllocator* m_localAllocator { nullptr };
};

} // namespace JSC

