/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
#include "IndexingType.h"

#include "JSCJSValueInlines.h"

namespace JSC {

IndexingType leastUpperBoundOfIndexingTypes(IndexingType a, IndexingType b)
{
    // It doesn't make sense to LUB something that is an array with something that isn't.
    ASSERT((a & IsArray) == (b & IsArray));

    // Boy, this sure is easy right now.
    return std::max(a, b);
}

IndexingType leastUpperBoundOfIndexingTypeAndType(IndexingType indexingType, SpeculatedType type)
{
    if (!type)
        return indexingType;
    switch (indexingType) {
    case ALL_BLANK_INDEXING_TYPES:
    case ALL_UNDECIDED_INDEXING_TYPES:
    case ALL_INT32_INDEXING_TYPES:
        if (isInt32Speculation(type))
            return (indexingType & ~IndexingShapeMask) | Int32Shape;
        // FIXME: Should this really say that it wants a double for NaNs.
        if (isFullNumberSpeculation(type))
            return (indexingType & ~IndexingShapeMask) | DoubleShape;
        return (indexingType & ~IndexingShapeMask) | ContiguousShape;
    case ALL_DOUBLE_INDEXING_TYPES:
        // FIXME: Should this really say that it wants a double for NaNs.
        if (isFullNumberSpeculation(type))
            return indexingType;
        return (indexingType & ~IndexingShapeMask) | ContiguousShape;
    case ALL_CONTIGUOUS_INDEXING_TYPES:
    case ALL_ARRAY_STORAGE_INDEXING_TYPES:
        return indexingType;
    default:
        CRASH();
        return 0;
    }
}

IndexingType leastUpperBoundOfIndexingTypeAndValue(IndexingType indexingType, JSValue value)
{
    return leastUpperBoundOfIndexingTypes(indexingType, indexingTypeForValue(value) | (indexingType & IsArray));
}

void dumpIndexingType(PrintStream& out, IndexingType indexingType)
{
    const char* basicName;
    switch (indexingType & AllArrayTypes) {
    case NonArray:
        basicName = "NonArray";
        break;
    case NonArrayWithInt32:
        basicName = "NonArrayWithInt32";
        break;
    case NonArrayWithDouble:
        basicName = "NonArrayWithDouble";
        break;
    case NonArrayWithContiguous:
        basicName = "NonArrayWithContiguous";
        break;
    case NonArrayWithArrayStorage:
        basicName = "NonArrayWithArrayStorage";
        break;
    case NonArrayWithSlowPutArrayStorage:
        basicName = "NonArrayWithSlowPutArrayStorage";
        break;
    case ArrayClass:
        basicName = "ArrayClass";
        break;
    case ArrayWithUndecided:
        basicName = "ArrayWithUndecided";
        break;
    case ArrayWithInt32:
        basicName = "ArrayWithInt32";
        break;
    case ArrayWithDouble:
        basicName = "ArrayWithDouble";
        break;
    case ArrayWithContiguous:
        basicName = "ArrayWithContiguous";
        break;
    case ArrayWithArrayStorage:
        basicName = "ArrayWithArrayStorage";
        break;
    case ArrayWithSlowPutArrayStorage:
        basicName = "ArrayWithSlowPutArrayStorage";
        break;
    case CopyOnWriteArrayWithInt32:
        basicName = "CopyOnWriteArrayWithInt32";
        break;
    case CopyOnWriteArrayWithDouble:
        basicName = "CopyOnWriteArrayWithDouble";
        break;
    case CopyOnWriteArrayWithContiguous:
        basicName = "CopyOnWriteArrayWithContiguous";
        break;
    default:
        basicName = "Unknown!";
        break;
    }
    
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    out.printf("%s%s", basicName, (indexingType & MayHaveIndexedAccessors) ? "|MayHaveIndexedAccessors" : "");
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
}

} // namespace JSC

