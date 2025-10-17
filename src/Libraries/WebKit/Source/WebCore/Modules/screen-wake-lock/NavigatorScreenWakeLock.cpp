/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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
#include "NavigatorScreenWakeLock.h"

#include "Navigator.h"
#include "WakeLock.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorScreenWakeLock);

NavigatorScreenWakeLock::NavigatorScreenWakeLock(Navigator& navigator)
    : m_navigator(navigator)
{
}

NavigatorScreenWakeLock::~NavigatorScreenWakeLock() = default;

NavigatorScreenWakeLock* NavigatorScreenWakeLock::from(Navigator& navigator)
{
    auto* supplement = static_cast<NavigatorScreenWakeLock*>(Supplement<Navigator>::from(&navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorScreenWakeLock>(navigator);
        supplement = newSupplement.get();
        provideTo(&navigator, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

ASCIILiteral NavigatorScreenWakeLock::supplementName()
{
    return "NavigatorScreenWakeLock"_s;
}

WakeLock& NavigatorScreenWakeLock::wakeLock(Navigator& navigator)
{
    return NavigatorScreenWakeLock::from(navigator)->wakeLock();
}

WakeLock& NavigatorScreenWakeLock::wakeLock()
{
    if (!m_wakeLock)
        m_wakeLock = WakeLock::create(downcast<Document>(m_navigator.scriptExecutionContext()));
    return *m_wakeLock;
}

} // namespace WebCore
