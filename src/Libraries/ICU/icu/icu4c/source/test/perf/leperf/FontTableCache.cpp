/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
#include "layout/LETypes.h"

//#include "letest.h"
#include "FontTableCache.h"

#define TABLE_CACHE_INIT 5
#define TABLE_CACHE_GROW 5

struct FontTableCacheEntry
{
  LETag tag;
  const void *table;
  size_t length;
};

FontTableCache::FontTableCache()
    : fTableCacheCurr(0), fTableCacheSize(TABLE_CACHE_INIT)
{
    fTableCache = LE_NEW_ARRAY(FontTableCacheEntry, fTableCacheSize);

    if (fTableCache == nullptr) {
        fTableCacheSize = 0;
        return;
    }

    for (int i = 0; i < fTableCacheSize; i += 1) {
        fTableCache[i].tag   = 0;
        fTableCache[i].table = nullptr;
        fTableCache[i].length = 0;
    }
}

FontTableCache::~FontTableCache()
{
    for (int i = fTableCacheCurr - 1; i >= 0; i -= 1) {
      LE_DELETE_ARRAY(fTableCache[i].table);

        fTableCache[i].tag   = 0;
        fTableCache[i].table = nullptr;
        fTableCache[i].length = 0;
    }

    fTableCacheCurr = 0;

    LE_DELETE_ARRAY(fTableCache);
}

void FontTableCache::freeFontTable(const void *table) const
{
  LE_DELETE_ARRAY(table);
}

const void *FontTableCache::find(LETag tableTag, size_t &length) const
{
    for (int i = 0; i < fTableCacheCurr; i += 1) {
        if (fTableCache[i].tag == tableTag) {
          length = fTableCache[i].length;
          return fTableCache[i].table;
        }
    }

    const void *table = readFontTable(tableTag, length);

    const_cast<FontTableCache*>(this)->add(tableTag, table, length);

    return table;
}

void FontTableCache::add(LETag tableTag, const void *table, size_t length)
{
    if (fTableCacheCurr >= fTableCacheSize) {
        le_int32 newSize = fTableCacheSize + TABLE_CACHE_GROW;

        fTableCache = static_cast<FontTableCacheEntry*>(LE_GROW_ARRAY(fTableCache, newSize));

        for (le_int32 i = fTableCacheSize; i < newSize; i += 1) {
            fTableCache[i].tag   = 0;
            fTableCache[i].table = nullptr;
            fTableCache[i].length = 0;
        }

        fTableCacheSize = newSize;
    }

    fTableCache[fTableCacheCurr].tag   = tableTag;
    fTableCache[fTableCacheCurr].table = table;
    fTableCache[fTableCacheCurr].length = length;

    fTableCacheCurr += 1;
}
