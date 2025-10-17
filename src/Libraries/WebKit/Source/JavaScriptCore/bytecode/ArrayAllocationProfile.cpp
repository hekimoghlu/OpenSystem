/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
#include "ArrayAllocationProfile.h"

#include "JSCellInlines.h"
#include <algorithm>

namespace JSC {

void ArrayAllocationProfile::updateProfile()
{
    // This is awkwardly racy but totally sound even when executed concurrently. The
    // worst cases go something like this:
    //
    // - Two threads race to execute this code; one of them succeeds in updating the
    //   m_currentIndexingType and the other either updates it again, or sees a null
    //   m_lastArray; if it updates it again then at worst it will cause the profile
    //   to "forget" some array. That's still sound, since we don't promise that
    //   this profile is a reflection of any kind of truth.
    //
    // - A concurrent thread reads m_lastArray, but that array is now dead. While
    //   it's possible for that array to no longer be reachable, it cannot actually
    //   be freed, since we require the GC to wait until all concurrent JITing
    //   finishes.
    //
    // But one exception is vector length. We access vector length to get the vector
    // length hint. However vector length can be accessible only from the main
    // thread because large butterfly can be realloced in the main thread.
    // So for now, we update the allocation profile only from the main thread.
    
    ASSERT(!isCompilationThread());
    Storage storage = std::exchange(m_storage, Storage(nullptr, m_storage.type()));
    JSArray* lastArray = storage.pointer();
    IndexingTypeAndVectorLength current = storage.type();
    if (!lastArray)
        return;
    if (LIKELY(Options::useArrayAllocationProfiling())) {
        // The basic model here is that we will upgrade ourselves to whatever the CoW version of lastArray is except ArrayStorage since we don't have CoW ArrayStorage.
        IndexingType indexingType = leastUpperBoundOfIndexingTypes(current.indexingType() & IndexingTypeMask, lastArray->indexingType());
        if (isCopyOnWrite(current.indexingType())) {
            if (indexingType > ArrayWithContiguous)
                indexingType = ArrayWithContiguous;
            indexingType |= CopyOnWrite;
        }
        unsigned largestSeenVectorLength = std::min(std::max(current.vectorLength(), lastArray->getVectorLength()), BASE_CONTIGUOUS_VECTOR_LEN_MAX);
        m_storage.setType(IndexingTypeAndVectorLength(indexingType, largestSeenVectorLength));
    }
}

} // namespace JSC

