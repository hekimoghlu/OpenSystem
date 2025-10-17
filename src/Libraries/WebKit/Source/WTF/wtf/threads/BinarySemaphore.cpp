/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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
#include <wtf/threads/BinarySemaphore.h>

namespace WTF {

void BinarySemaphore::signal()
{
    Locker locker { m_lock };
    m_isSet = true;
    m_condition.notifyOne();
}

bool BinarySemaphore::waitUntil(const TimeWithDynamicClockType& absoluteTime)
{
    Locker locker { m_lock };
    bool satisfied = m_condition.waitUntil(m_lock, absoluteTime, [&] {
        assertIsHeld(m_lock);
        return m_isSet;
    });
    if (satisfied)
        m_isSet = false;
    return satisfied;
}

} // namespace WTF
