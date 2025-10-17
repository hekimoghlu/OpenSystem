/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// BlobCache_unittest.h: Unit tests for the blob cache.

#include <gtest/gtest.h>

#include "libANGLE/BlobCache.h"

namespace egl
{

// Note: this is fairly similar to SizedMRUCache_unittest, and makes sure the
// BlobCache usage of SizedMRUCache is not broken.

using BlobPut = angle::MemoryBuffer;
using Blob    = BlobCache::Value;
using Key     = BlobCache::Key;

template <typename T>
void MakeSequence(T &seq, uint8_t start)
{
    for (uint8_t i = 0; i < seq.size(); ++i)
    {
        seq[i] = i + start;
    }
}

BlobPut MakeBlob(size_t size, uint8_t start = 0)
{
    BlobPut blob;
    EXPECT_TRUE(blob.resize(size));
    MakeSequence(blob, start);
    return blob;
}

Key MakeKey(uint8_t start = 0)
{
    Key key;
    MakeSequence(key, start);
    return key;
}

// Test a cache with a value that takes up maximum size.
TEST(BlobCacheTest, MaxSizedValue)
{
    constexpr size_t kSize = 32;
    BlobCache blobCache(kSize);

    blobCache.populate(MakeKey(0), MakeBlob(kSize));
    EXPECT_EQ(32u, blobCache.size());
    EXPECT_FALSE(blobCache.empty());

    blobCache.populate(MakeKey(1), MakeBlob(kSize));
    EXPECT_EQ(32u, blobCache.size());
    EXPECT_FALSE(blobCache.empty());

    Blob blob;
    EXPECT_FALSE(blobCache.get(nullptr, nullptr, MakeKey(0), &blob));

    blobCache.clear();
    EXPECT_TRUE(blobCache.empty());
}

// Test a cache with many small values, that it can handle unlimited inserts.
TEST(BlobCacheTest, ManySmallValues)
{
    constexpr size_t kSize = 32;
    BlobCache blobCache(kSize);

    for (size_t value = 0; value < kSize; ++value)
    {
        blobCache.populate(MakeKey(value), MakeBlob(1, value));

        Blob qvalue;
        EXPECT_TRUE(blobCache.get(nullptr, nullptr, MakeKey(value), &qvalue));
        if (qvalue.size() > 0)
        {
            EXPECT_EQ(value, qvalue[0]);
        }
    }

    EXPECT_EQ(32u, blobCache.size());
    EXPECT_FALSE(blobCache.empty());

    // Putting one element evicts the first element.
    blobCache.populate(MakeKey(kSize), MakeBlob(1, kSize));

    Blob qvalue;
    EXPECT_FALSE(blobCache.get(nullptr, nullptr, MakeKey(0), &qvalue));

    // Putting one large element cleans out the whole stack.
    blobCache.populate(MakeKey(kSize + 1), MakeBlob(kSize, kSize + 1));
    EXPECT_EQ(32u, blobCache.size());
    EXPECT_FALSE(blobCache.empty());

    for (size_t value = 0; value <= kSize; ++value)
    {
        EXPECT_FALSE(blobCache.get(nullptr, nullptr, MakeKey(value), &qvalue));
    }
    EXPECT_TRUE(blobCache.get(nullptr, nullptr, MakeKey(kSize + 1), &qvalue));
    if (qvalue.size() > 0)
    {
        EXPECT_EQ(kSize + 1, qvalue[0]);
    }

    // Put a bunch of items in the cache sequentially.
    for (size_t value = 0; value < kSize * 10; ++value)
    {
        blobCache.populate(MakeKey(value), MakeBlob(1, value));
    }

    EXPECT_EQ(32u, blobCache.size());
}

// Tests putting an oversize element.
TEST(BlobCacheTest, OversizeValue)
{
    constexpr size_t kSize = 32;
    BlobCache blobCache(kSize);

    blobCache.populate(MakeKey(5), MakeBlob(100));

    Blob qvalue;
    EXPECT_FALSE(blobCache.get(nullptr, nullptr, MakeKey(5), &qvalue));
}

}  // namespace egl
