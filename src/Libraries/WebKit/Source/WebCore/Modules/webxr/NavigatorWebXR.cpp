/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#include "NavigatorWebXR.h"

#if ENABLE(WEBXR)

#include "Navigator.h"
#include "WebXRSystem.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorWebXR);

WebXRSystem& NavigatorWebXR::xr(Navigator& navigatorObject)
{
    auto& navigator = NavigatorWebXR::from(navigatorObject);
    if (!navigator.m_xr)
        navigator.m_xr = WebXRSystem::create(navigatorObject);
    return *navigator.m_xr;
}

WebXRSystem* NavigatorWebXR::xrIfExists(Navigator& navigator)
{
    return NavigatorWebXR::from(navigator).m_xr.get();
}

NavigatorWebXR& NavigatorWebXR::from(Navigator& navigator)
{
    auto* supplement = static_cast<NavigatorWebXR*>(Supplement<Navigator>::from(&navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorWebXR>();
        supplement = newSupplement.get();
        provideTo(&navigator, supplementName(), WTFMove(newSupplement));
    }
    return *supplement;
}

ASCIILiteral NavigatorWebXR::supplementName()
{
    return "NavigatorWebXR"_s;
}

} // namespace WebCore

#endif // ENABLE(WEBXR)
