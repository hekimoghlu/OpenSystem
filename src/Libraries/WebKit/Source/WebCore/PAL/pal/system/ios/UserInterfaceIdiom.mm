/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#import "config.h"
#import "UserInterfaceIdiom.h"

#if PLATFORM(IOS_FAMILY)

#import "Device.h"
#import <pal/spi/ios/UIKitSPI.h>

#if USE(APPLE_INTERNAL_SDK)
#import <WebKitAdditions/UserInterfaceIdiomAdditionsBefore.mm>
#endif

#import <pal/ios/UIKitSoftLink.h>

namespace PAL {

static std::atomic<std::optional<UserInterfaceIdiom>> s_currentUserInterfaceIdiom;

#if USE(APPLE_INTERNAL_SDK)
#import <WebKitAdditions/UserInterfaceIdiomAdditionsAfter.mm>
#else
static bool shouldForceUserInterfaceIdiomSmallScreen(std::optional<UIUserInterfaceIdiom> = std::nullopt)
{
    return false;
}
#endif

bool currentUserInterfaceIdiomIsSmallScreen()
{
    if (!s_currentUserInterfaceIdiom.load())
        updateCurrentUserInterfaceIdiom();
    return s_currentUserInterfaceIdiom.load() == UserInterfaceIdiom::SmallScreen;
}

bool currentUserInterfaceIdiomIsVision()
{
    if (!s_currentUserInterfaceIdiom.load())
        updateCurrentUserInterfaceIdiom();
    auto idiom = *s_currentUserInterfaceIdiom.load();
    return idiom == UserInterfaceIdiom::Vision;
}

UserInterfaceIdiom currentUserInterfaceIdiom()
{
    if (!s_currentUserInterfaceIdiom.load())
        updateCurrentUserInterfaceIdiom();
    return s_currentUserInterfaceIdiom.load().value_or(UserInterfaceIdiom::Default);
}

void setCurrentUserInterfaceIdiom(UserInterfaceIdiom idiom)
{
    s_currentUserInterfaceIdiom = idiom;
}

bool updateCurrentUserInterfaceIdiom()
{
    UserInterfaceIdiom oldIdiom = s_currentUserInterfaceIdiom.load().value_or(UserInterfaceIdiom::Default);

    // If we are in a daemon, we cannot use UIDevice. Fall back to checking the hardware itself.
    // Since daemons don't ever run in an iPhone-app-on-iPad jail, this will be accurate in the daemon case,
    // but is not sufficient in the application case.
    UserInterfaceIdiom newIdiom = [&] {
        if (![PAL::getUIApplicationClass() sharedApplication]) {
            if (PAL::deviceClassIsSmallScreen() || shouldForceUserInterfaceIdiomSmallScreen())
                return UserInterfaceIdiom::SmallScreen;
            if (PAL::deviceClassIsVision())
                return UserInterfaceIdiom::Vision;
        } else {
            auto idiom = [[PAL::getUIDeviceClass() currentDevice] userInterfaceIdiom];
            if (idiom == UIUserInterfaceIdiomPhone || idiom == UIUserInterfaceIdiomWatch || shouldForceUserInterfaceIdiomSmallScreen(idiom))
                return UserInterfaceIdiom::SmallScreen;
#if HAVE(UI_USER_INTERFACE_IDIOM_VISION)
            if (idiom == UIUserInterfaceIdiomVision)
                return UserInterfaceIdiom::Vision;
#endif
        }

        return UserInterfaceIdiom::Default;
    }();

    if (s_currentUserInterfaceIdiom.load() && oldIdiom == newIdiom)
        return false;

    setCurrentUserInterfaceIdiom(newIdiom);
    return true;
}

} // namespace PAL

#endif
