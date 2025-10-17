/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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

#if ENABLE(GEOLOCATION)

#include "NavigatorGeolocation.h"

#include "Document.h"
#include "Geolocation.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "Navigator.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorGeolocation);

NavigatorGeolocation::NavigatorGeolocation(Navigator& navigator)
    : m_navigator(navigator)
{
}

NavigatorGeolocation::~NavigatorGeolocation() = default;

ASCIILiteral NavigatorGeolocation::supplementName()
{
    return "NavigatorGeolocation"_s;
}

NavigatorGeolocation* NavigatorGeolocation::from(Navigator& navigator)
{
    NavigatorGeolocation* supplement = static_cast<NavigatorGeolocation*>(Supplement<Navigator>::from(&navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorGeolocation>(navigator);
        supplement = newSupplement.get();
        provideTo(&navigator, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

#if PLATFORM(IOS_FAMILY)
void NavigatorGeolocation::resetAllGeolocationPermission()
{
    if (m_geolocation)
        m_geolocation->resetAllGeolocationPermission();
}
#endif // PLATFORM(IOS_FAMILY)

Geolocation* NavigatorGeolocation::geolocation(Navigator& navigator)
{
    return NavigatorGeolocation::from(navigator)->geolocation();
}

Geolocation* NavigatorGeolocation::geolocation() const
{
    if (!m_geolocation)
        m_geolocation = Geolocation::create(Ref { m_navigator.get() });
    return m_geolocation.get();
}

} // namespace WebCore

#endif // ENABLE(GEOLOCATION)
