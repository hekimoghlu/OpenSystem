/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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

#include <wtf/FastMalloc.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class MallocStatistics : public RefCounted<MallocStatistics> {
public:
    static Ref<MallocStatistics> create() { return adoptRef(*new MallocStatistics); }

    size_t reservedVMBytes() const { return m_stats.reservedVMBytes; }
    size_t committedVMBytes() const { return m_stats.committedVMBytes; }
    size_t freeListBytes() const { return m_stats.freeListBytes; }

private:
    MallocStatistics()
        : m_stats(WTF::fastMallocStatistics())
    {
    }

    WTF::FastMallocStatistics m_stats;
};

} // namespace WebCore
