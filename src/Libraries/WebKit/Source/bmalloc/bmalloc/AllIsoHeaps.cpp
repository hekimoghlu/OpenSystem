/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
#include "BPlatform.h"
#include "AllIsoHeaps.h"

#if !BUSE(TZONE)
#if !BUSE(LIBPAS)

namespace bmalloc {

DEFINE_STATIC_PER_PROCESS_STORAGE(AllIsoHeaps);

AllIsoHeaps::AllIsoHeaps(const LockHolder&)
{
}

void AllIsoHeaps::add(IsoHeapImplBase* heap)
{
    LockHolder locker(mutex());
    heap->m_next = m_head;
    m_head = heap;
}

IsoHeapImplBase* AllIsoHeaps::head()
{
    LockHolder locker(mutex());
    return m_head;
}

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
