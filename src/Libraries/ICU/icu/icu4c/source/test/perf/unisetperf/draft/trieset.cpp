/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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
#include "cmemory.h"
#include "unicode/uniset.h"
#include "unicode/uobject.h"
#include "unicode/usetiter.h"
#include "unicode/utypes.h"
#include "unicont.h"
#include "utrie.h"

using icu::UObject;
using icu::UnicodeSet;
using icu::UnicodeSetIterator;

#define UTRIE_GET8_LATIN1(trie) ((const uint8_t *)(trie)->data32+UTRIE_DATA_BLOCK_LENGTH)

#define UTRIE_GET8_FROM_LEAD(trie, c16) \
    ((const uint8_t *)(trie)->data32)[ \
        ((int32_t)((trie)->index[(c16)>>UTRIE_SHIFT])<<UTRIE_INDEX_SHIFT)+ \
        ((c16)&UTRIE_MASK) \
    ]

class TrieSet : public UObject, public UnicodeContainable {
public:
    TrieSet(const UnicodeSet &set, UErrorCode &errorCode)
            : trieData(nullptr), latin1(nullptr), restSet(set.clone()) {
        if(U_FAILURE(errorCode)) {
            return;
        }
        if(restSet==nullptr) {
            errorCode=U_MEMORY_ALLOCATION_ERROR;
            return;
        }

        UNewTrie *newTrie=utrie_open(nullptr, nullptr, 0x11000, 0, 0, true);
        UChar32 start, end;

        UnicodeSetIterator iter(set);

        while(iter.nextRange() && !iter.isString()) {
            start=iter.getCodepoint();
            end=iter.getCodepointEnd();
            if(start>0xffff) {
                break;
            }
            if(end>0xffff) {
                end=0xffff;
            }
            if(!utrie_setRange32(newTrie, start, end+1, true, true)) {
                errorCode=U_INTERNAL_PROGRAM_ERROR;
                return;
            }
        }

        // Preflight the trie length.
        int32_t length=utrie_serialize(newTrie, nullptr, 0, nullptr, 8, &errorCode);
        if(errorCode!=U_BUFFER_OVERFLOW_ERROR) {
            return;
        }

        trieData = static_cast<uint32_t*>(uprv_malloc(length));
        if(trieData==nullptr) {
            errorCode=U_MEMORY_ALLOCATION_ERROR;
            return;
        }

        errorCode=U_ZERO_ERROR;
        utrie_serialize(newTrie, trieData, length, nullptr, 8, &errorCode);
        utrie_unserialize(&trie, trieData, length, &errorCode);  // TODO: Implement for 8-bit UTrie!

        if(U_SUCCESS(errorCode)) {
            // Copy the indexes for surrogate code points into the BMP range
            // for simple access across the entire BMP.
            uprv_memcpy((uint16_t *)trie.index+(0xd800>>UTRIE_SHIFT),
                        trie.index+UTRIE_BMP_INDEX_LENGTH,
                        (0x800>>UTRIE_SHIFT)*2);
            latin1=UTRIE_GET8_LATIN1(&trie);
        }

        restSet->remove(0, 0xffff);
    }

    ~TrieSet() {
        uprv_free(trieData);
        delete restSet;
    }

    UBool contains(UChar32 c) const override {
        if (static_cast<uint32_t>(c) <= 0xff) {
            return latin1[c];
        } else if (static_cast<uint32_t>(c) < 0xffff) {
            return UTRIE_GET8_FROM_LEAD(&trie, c);
        } else {
            return restSet->contains(c);
        }
    }

private:
    uint32_t *trieData;
    const uint8_t *latin1;
    UTrie trie;
    UnicodeSet *restSet;
};
