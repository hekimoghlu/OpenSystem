/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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

#include "IsoDeallocator.h"
#include "IsoTLSEntry.h"
#include "Mutex.h"
#include <mutex>

namespace bmalloc {

template<typename Config>
class IsoTLSDeallocatorEntry : public DefaultIsoTLSEntry<IsoDeallocator<Config>> {
public:
    IsoTLSDeallocatorEntry(const std::lock_guard<Mutex>&);
    ~IsoTLSDeallocatorEntry();
    
    // This is used as the heap lock, since heaps in the same size class share the same deallocator
    // logs.
    Mutex lock;
    
private:
    void construct(void* entry) override;
};

} // namespace bmalloc

