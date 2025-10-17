/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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
#include "IsoDirectory.h"

namespace bmalloc {

template<typename Config>
class IsoDirectoryPage {
    MAKE_BMALLOCED;
public:
    // By my math, this results in an IsoDirectoryPage using 4036 bytes on 64-bit platforms. We allocate it
    // with malloc, so it doesn't really matter how much memory it uses. But it's nice that this about a
    // page, since that's quite intuitive.
    //
    // On 32-bit platforms, I think that this will be 2112 bytes. That's still quite intuitive.
    //
    // Note that this is a multiple of 32 so that it uses bitvectors efficiently.
    static constexpr unsigned numPages = 480;
    
    IsoDirectoryPage(IsoHeapImpl<Config>&, unsigned index);
    
    static IsoDirectoryPage* pageFor(IsoDirectory<Config, numPages>* payload);
    
    unsigned index() const { return m_index; }
    
    IsoDirectory<Config, numPages> payload;
    IsoDirectoryPage* next { nullptr };

private:
    unsigned m_index;
};

} // namespace bmalloc

