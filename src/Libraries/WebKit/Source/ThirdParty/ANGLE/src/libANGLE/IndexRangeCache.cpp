/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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

//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// IndexRangeCache.cpp: Defines the gl::IndexRangeCache class which stores information about
// ranges of indices.

#include "libANGLE/IndexRangeCache.h"

#include "common/debug.h"
#include "libANGLE/formatutils.h"

namespace gl
{

IndexRangeCache::IndexRangeCache() {}

IndexRangeCache::~IndexRangeCache() {}

void IndexRangeCache::addRange(DrawElementsType type,
                               size_t offset,
                               size_t count,
                               bool primitiveRestartEnabled,
                               const IndexRange &range)
{
    mIndexRangeCache[IndexRangeKey(type, offset, count, primitiveRestartEnabled)] = range;
}

bool IndexRangeCache::findRange(DrawElementsType type,
                                size_t offset,
                                size_t count,
                                bool primitiveRestartEnabled,
                                IndexRange *outRange) const
{
    auto i = mIndexRangeCache.find(IndexRangeKey(type, offset, count, primitiveRestartEnabled));
    if (i != mIndexRangeCache.end())
    {
        if (outRange)
        {
            *outRange = i->second;
        }
        return true;
    }
    else
    {
        if (outRange)
        {
            *outRange = IndexRange();
        }
        return false;
    }
}

void IndexRangeCache::invalidateRange(size_t offset, size_t size)
{
    size_t invalidateStart = offset;
    size_t invalidateEnd   = offset + size;

    auto i = mIndexRangeCache.begin();
    while (i != mIndexRangeCache.end())
    {
        size_t rangeStart = i->first.offset;
        size_t rangeEnd =
            i->first.offset + (GetDrawElementsTypeSize(i->first.type) * i->first.count);

        if (invalidateEnd < rangeStart || invalidateStart > rangeEnd)
        {
            ++i;
        }
        else
        {
            mIndexRangeCache.erase(i++);
        }
    }
}

void IndexRangeCache::clear()
{
    mIndexRangeCache.clear();
}

IndexRangeCache::IndexRangeKey::IndexRangeKey()
    : IndexRangeCache::IndexRangeKey(DrawElementsType::InvalidEnum, 0, 0, false)
{}

IndexRangeCache::IndexRangeKey::IndexRangeKey(DrawElementsType type_,
                                              size_t offset_,
                                              size_t count_,
                                              bool primitiveRestartEnabled_)
    : type(type_), offset(offset_), count(count_), primitiveRestartEnabled(primitiveRestartEnabled_)
{}

bool IndexRangeCache::IndexRangeKey::operator<(const IndexRangeKey &rhs) const
{
    if (type != rhs.type)
    {
        return type < rhs.type;
    }
    if (offset != rhs.offset)
    {
        return offset < rhs.offset;
    }
    if (count != rhs.count)
    {
        return count < rhs.count;
    }
    if (primitiveRestartEnabled != rhs.primitiveRestartEnabled)
    {
        return primitiveRestartEnabled;
    }
    return false;
}

}  // namespace gl
