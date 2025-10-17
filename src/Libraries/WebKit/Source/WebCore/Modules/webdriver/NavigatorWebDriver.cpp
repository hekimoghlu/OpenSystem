/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#include "NavigatorWebDriver.h"

#include "LocalFrame.h"
#include "Navigator.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorWebDriver);

NavigatorWebDriver::NavigatorWebDriver() = default;

NavigatorWebDriver::~NavigatorWebDriver() = default;

ASCIILiteral NavigatorWebDriver::supplementName()
{
    return "NavigatorWebDriver"_s;
}

bool NavigatorWebDriver::isControlledByAutomation(const Navigator& navigator)
{
    RefPtr frame = navigator.frame();
    if (!frame || !frame->page())
        return false;

    return frame->page()->isControlledByAutomation();
}

NavigatorWebDriver* NavigatorWebDriver::from(Navigator* navigator)
{
    NavigatorWebDriver* supplement = static_cast<NavigatorWebDriver*>(Supplement<Navigator>::from(navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorWebDriver>();
        supplement = newSupplement.get();
        provideTo(navigator, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

bool NavigatorWebDriver::webdriver(const Navigator& navigator)
{
    return NavigatorWebDriver::isControlledByAutomation(navigator);
}

} // namespace WebCore
