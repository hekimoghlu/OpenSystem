/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
#include "config.h"
#include "WasmContext.h"

#if ENABLE(WEBASSEMBLY)

namespace JSC { namespace Wasm {

uint64_t* Context::scratchBufferForSize(size_t size)
{
    if (!size)
        return nullptr;

    Locker locker { m_scratchBufferLock };
    if (size > m_sizeOfLastScratchBuffer) {
        m_sizeOfLastScratchBuffer = size * 2;

        auto newBuffer = makeUniqueArray<uint64_t>(m_sizeOfLastScratchBuffer);
        RELEASE_ASSERT(newBuffer);
        m_scratchBuffers.append(WTFMove(newBuffer));
    }
    // Scanning scratch buffers for GC is not necessary since while performing OSR entry, we do not perform GC.
    return m_scratchBuffers.last().get();
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
