/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
#include "SWOriginStore.h"

#include "SecurityOrigin.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SWOriginStore);

void SWOriginStore::add(const SecurityOriginData& origin)
{
    ++m_originCounts.ensure(origin, [&] {
        addToStore(origin);
        return 0;
    }).iterator->value;
}

void SWOriginStore::remove(const SecurityOriginData& origin)
{
    auto iterator = m_originCounts.find(origin);
    ASSERT(iterator != m_originCounts.end());
    if (iterator == m_originCounts.end())
        return;

    if (--iterator->value)
        return;

    m_originCounts.remove(iterator);
    removeFromStore(origin);
}

void SWOriginStore::clear(const SecurityOriginData& origin)
{
    m_originCounts.remove(origin);
    removeFromStore(origin);
}

void SWOriginStore::clearAll()
{
    m_originCounts.clear();
    clearStore();
}

} // namespace WebCore
