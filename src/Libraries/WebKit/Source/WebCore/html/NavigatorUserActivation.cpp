/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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
#include "NavigatorUserActivation.h"

#include "Navigator.h"
#include "UserActivation.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorUserActivation);

NavigatorUserActivation::NavigatorUserActivation(Navigator& navigator)
    : m_userActivation(UserActivation::create(navigator))
{
}

NavigatorUserActivation::~NavigatorUserActivation() = default;

Ref<UserActivation> NavigatorUserActivation::userActivation(Navigator& navigator)
{
    return NavigatorUserActivation::from(navigator)->userActivation();
}

Ref<UserActivation> NavigatorUserActivation::userActivation()
{
    return m_userActivation;
}

NavigatorUserActivation* NavigatorUserActivation::from(Navigator& navigator)
{
    auto* supplement = static_cast<NavigatorUserActivation*>(Supplement<Navigator>::from(&navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorUserActivation>(navigator);
        supplement = newSupplement.get();
        provideTo(&navigator, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

ASCIILiteral NavigatorUserActivation::supplementName()
{
    return "NavigatorUserActivation"_s;
}

}
