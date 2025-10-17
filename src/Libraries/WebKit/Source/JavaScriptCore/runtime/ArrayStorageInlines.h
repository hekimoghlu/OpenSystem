/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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

#include "ArrayStorage.h"
#include "StructureInlines.h"

namespace JSC {

inline unsigned ArrayStorage::availableVectorLength(unsigned indexBias, Structure* structure, unsigned vectorLength)
{
    return availableVectorLength(indexBias, structure->outOfLineCapacity(), vectorLength);
}

inline unsigned ArrayStorage::availableVectorLength(size_t propertyCapacity, unsigned vectorLength)
{
    return availableVectorLength(m_indexBias, propertyCapacity, vectorLength);
}

inline unsigned ArrayStorage::availableVectorLength(Structure* structure, unsigned vectorLength)
{
    return availableVectorLength(structure->outOfLineCapacity(), vectorLength);
}

inline unsigned ArrayStorage::optimalVectorLength(unsigned indexBias, size_t propertyCapacity, unsigned vectorLength)
{
    vectorLength = std::max(BASE_ARRAY_STORAGE_VECTOR_LEN, vectorLength);
    return availableVectorLength(indexBias, propertyCapacity, vectorLength);
}

inline unsigned ArrayStorage::optimalVectorLength(unsigned indexBias, Structure* structure, unsigned vectorLength)
{
    return optimalVectorLength(indexBias, structure->outOfLineCapacity(), vectorLength);
}

inline unsigned ArrayStorage::optimalVectorLength(size_t propertyCapacity, unsigned vectorLength)
{
    return optimalVectorLength(m_indexBias, propertyCapacity, vectorLength);
}

inline unsigned ArrayStorage::optimalVectorLength(Structure* structure, unsigned vectorLength)
{
    return optimalVectorLength(structure->outOfLineCapacity(), vectorLength);
}

inline size_t ArrayStorage::totalSize(Structure* structure) const
{
    return totalSize(structure->outOfLineCapacity());
}

} // namespace JSC
