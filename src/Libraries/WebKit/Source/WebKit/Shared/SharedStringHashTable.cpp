/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#include "SharedStringHashTable.h"

#include <WebCore/SharedMemory.h>
#include <wtf/StdLibExtras.h>

namespace WebKit {

using namespace WebCore;

SharedStringHashTable::SharedStringHashTable()
{
}

SharedStringHashTable::~SharedStringHashTable()
{
}

bool SharedStringHashTable::add(SharedStringHash sharedStringHash)
{
    auto* slot = findSlot(sharedStringHash);
    ASSERT(slot);

    // Check if the same link hash is in the table already.
    if (*slot)
        return false;

    *slot = sharedStringHash;
    return true;
}

bool SharedStringHashTable::remove(SharedStringHash sharedStringHash)
{
    auto* slot = findSlot(sharedStringHash);
    if (!slot || !*slot)
        return false;

    *slot = 0;
    return true;
}

void SharedStringHashTable::clear()
{
    if (!m_sharedMemory)
        return;

    zeroSpan(m_sharedMemory->mutableSpan());
    setSharedMemory(nullptr);
}

} // namespace WebKit
