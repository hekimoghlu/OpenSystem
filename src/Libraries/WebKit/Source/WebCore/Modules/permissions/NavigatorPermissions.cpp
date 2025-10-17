/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 3, 2025.
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
#include "NavigatorPermissions.h"

#include "Navigator.h"
#include "Permissions.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorPermissions);

NavigatorPermissions::NavigatorPermissions(Navigator& navigator)
    : m_navigator(navigator)
{
}

Permissions& NavigatorPermissions::permissions(Navigator& navigator)
{
    return NavigatorPermissions::from(navigator).permissions();
}

Permissions& NavigatorPermissions::permissions()
{
    if (!m_permissions)
        m_permissions = Permissions::create(m_navigator);

    return *m_permissions;
}

NavigatorPermissions& NavigatorPermissions::from(Navigator& navigator)
{
    auto* supplement = static_cast<NavigatorPermissions*>(Supplement<Navigator>::from(&navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorPermissions>(navigator);
        supplement = newSupplement.get();
        provideTo(&navigator, supplementName(), WTFMove(newSupplement));
    }

    return *supplement;
}

ASCIILiteral NavigatorPermissions::supplementName()
{
    return "NavigatorPermissions"_s;
}

} // namespace WebCore
