/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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
#include "SleepDisablerCocoa.h"

#if PLATFORM(COCOA)
#include <pal/spi/cocoa/IOPMLibSPI.h>
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

namespace PAL {

std::unique_ptr<SleepDisabler> SleepDisabler::create(const String& reason, Type type)
{
    return std::unique_ptr<SleepDisabler>(new SleepDisablerCocoa(reason, type));
}

SleepDisablerCocoa::SleepDisablerCocoa(const String& reason, Type type)
    : SleepDisabler(reason, type)
{
    switch (type) {
    case Type::Display:
        takeScreenSleepDisablingAssertion(reason);
        break;
    case Type::System:
        takeSystemSleepDisablingAssertion(reason);
        break;
    default:
        ASSERT_NOT_REACHED();
        break;
    }
}

SleepDisablerCocoa::~SleepDisablerCocoa()
{
#if PLATFORM(IOS_FAMILY)
    m_screenSleepDisablerToken = nullptr;
#endif
    if (m_sleepAssertion)
        IOPMAssertionRelease(m_sleepAssertion);
}

#if PLATFORM(MAC)
void SleepDisablerCocoa::takeScreenSleepDisablingAssertion(const String& reason)
{
    IOPMAssertionCreateWithDescription(kIOPMAssertionTypePreventUserIdleDisplaySleep, reason.createCFString().get(), nullptr, nullptr, nullptr, 0, nullptr, &m_sleepAssertion);
}
#endif

void SleepDisablerCocoa::takeSystemSleepDisablingAssertion(const String& reason)
{
    IOPMAssertionCreateWithDescription(kIOPMAssertionTypePreventUserIdleSystemSleep, reason.createCFString().get(), nullptr, nullptr, nullptr, 0, nullptr, &m_sleepAssertion);
}

} // namespace PAL

#endif // PLATFORM(COCOA)
