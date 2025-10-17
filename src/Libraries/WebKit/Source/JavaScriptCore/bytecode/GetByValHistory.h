/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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

#include "TinyBloomFilter.h"
#include <wtf/text/UniquedStringImpl.h>

namespace JSC {

struct GetByValHistory {
    void observeNonUID()
    {
        uintptr_t count = Options::getByValICMaxNumberOfIdentifiers() + 1;
        update(count, filter());
    }

    void observe(const UniquedStringImpl* impl)
    {
        if (!impl) {
            observeNonUID();
            return;
        }

        uintptr_t count = this->count();
        uintptr_t filter = this->filter();

        TinyBloomFilter<uintptr_t> bloomFilter(filter);
        uintptr_t implBits = std::bit_cast<uintptr_t>(impl);
        ASSERT(((static_cast<uint64_t>(implBits) << 8) >> 8) == static_cast<uint64_t>(implBits));
        if (bloomFilter.ruleOut(implBits)) {
            bloomFilter.add(implBits);
            ++count;
            update(count, bloomFilter.bits());
        }
    }

    uintptr_t count() const { return static_cast<uintptr_t>(m_payload >> 56); }

private:
    uintptr_t filter() const { return static_cast<uintptr_t>((m_payload << 8) >> 8); }

    void update(uint64_t count, uint64_t filter)
    {
        ASSERT(((filter << 8) >> 8) == filter);
        m_payload = (count << 56) | filter;
    }

    uint64_t m_payload { 0 };
};

} // namespace JSC
