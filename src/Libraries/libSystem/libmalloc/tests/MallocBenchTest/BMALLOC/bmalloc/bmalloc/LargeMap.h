/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
#ifndef LargeMap_h
#define LargeMap_h

#include "LargeRange.h"
#include "Vector.h"
#include <algorithm>

namespace bmalloc {

class LargeMap {
public:
    LargeRange* begin() { return m_free.begin(); }
    LargeRange* end() { return m_free.end(); }

    void add(const LargeRange&);
    LargeRange remove(size_t alignment, size_t);
    Vector<LargeRange>& ranges() { return m_free; }
    void markAllAsEligibile();

    size_t size() { return m_free.size(); }
    LargeRange& at(size_t i) { return m_free[i]; }

private:
    Vector<LargeRange> m_free;
};

} // namespace bmalloc

#endif // LargeMap_h
