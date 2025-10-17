/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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

#include "IsoHeap.h"
#include "IsoPage.h"
#include "IsoSharedConfig.h"

#if !BUSE(LIBPAS)

namespace bmalloc {

class IsoHeapImplBase;

class IsoSharedPage : public IsoPageBase {
public:
    BEXPORT static IsoSharedPage* tryCreate();

    template<typename Config, typename Type>
    void free(const LockHolder&, api::IsoHeapBase<Type>&, void*);
    VariadicBumpAllocator startAllocating(const LockHolder&);
    void stopAllocating(const LockHolder&);

private:
    IsoSharedPage()
        : IsoPageBase(true)
    {
    }
};

template<typename Config>
uint8_t* indexSlotFor(void* ptr)
{
    BASSERT(IsoPageBase::pageFor(ptr)->isShared());
    return static_cast<uint8_t*>(ptr) + Config::objectSize;
}

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
