/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#include "ObjectType.h"

#include "Chunk.h"
#include "Heap.h"
#include "Object.h"
#include "PerProcess.h"

namespace bmalloc {

ObjectType objectType(HeapKind kind, void* object)
{
    if (mightBeLarge(object)) {
        if (!object)
            return ObjectType::Small;

        std::unique_lock<Mutex> lock(Heap::mutex());
        if (PerProcess<PerHeapKind<Heap>>::getFastCase()->at(kind).isLarge(lock, object))
            return ObjectType::Large;
    }
    
    return ObjectType::Small;
}

} // namespace bmalloc
