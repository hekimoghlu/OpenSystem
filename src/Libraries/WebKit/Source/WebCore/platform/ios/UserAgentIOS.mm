/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
#import "UserAgent.h"

#if PLATFORM(IOS_FAMILY)

#import "SystemVersion.h"
#import <pal/spi/ios/MobileGestaltSPI.h>
#import <pal/spi/ios/UIKitSPI.h>
#import <pal/system/ios/Device.h>
#import <wtf/RetainPtr.h>
#import <wtf/RuntimeApplicationChecks.h>
#import <wtf/cf/TypeCastsCF.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#import <wtf/text/MakeString.h>

#import <pal/ios/UIKitSoftLink.h>

namespace WebCore {

static inline bool isClassic()
{
    return [[PAL::getUIApplicationClass() sharedApplication] _isClassic];
}

static inline bool isClassicPad()
{
    return [PAL::getUIApplicationClass() _classicMode] == UIApplicationSceneClassicModeOriginalPad;
}

static inline bool isClassicPhone()
{
    return isClassic() && [PAL::getUIApplicationClass() _classicMode] != UIApplicationSceneClassicModeOriginalPad;
}

ASCIILiteral osNameForUserAgent()
{
    if (PAL::deviceHasIPadCapability() && !isClassicPhone())
        return "OS"_s;
    return "iPhone OS"_s;
}

#if !ENABLE(STATIC_IPAD_USER_AGENT_VALUE)
static StringView deviceNameForUserAgent()
{
    if (isClassic()) {
        if (isClassicPad())
            return "iPad"_s;
        return "iPhone"_s;
    }

    static NeverDestroyed<String> name = [] {
        auto name = PAL::deviceName();
#if PLATFORM(IOS_FAMILY_SIMULATOR)
        size_t location = name.find(" Simulator"_s);
        if (location != notFound)
            return name.left(location);
#endif
        return name;
    }();
    return name.get();
}
#endif

String standardUserAgentWithApplicationName(const String& applicationName, const String& userAgentOSVersion, UserAgentType type)
{
    auto separator = applicationName.isEmpty() ? ""_s : " "_s;

    if (type == UserAgentType::Desktop)
        return makeString("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)"_s, separator, applicationName);

#if ENABLE(STATIC_IPAD_USER_AGENT_VALUE)
    UNUSED_PARAM(userAgentOSVersion);
    return makeString("Mozilla/5.0 (iPad; CPU OS 16_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko)"_s, separator, applicationName);
#else
    if (!linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::DoesNotOverrideUAFromNSUserDefault)) {
        if (auto override = dynamic_cf_cast<CFStringRef>(adoptCF(CFPreferencesCopyAppValue(CFSTR("UserAgent"), CFSTR("com.apple.WebFoundation"))))) {
            static BOOL hasLoggedDeprecationWarning = NO;
            if (!hasLoggedDeprecationWarning) {
                NSLog(@"Reading an override UA from the NSUserDefault [com.apple.WebFoundation UserAgent]. This is incompatible with the modern need to compose the UA and clients should use the API to set the application name or UA instead.");
                hasLoggedDeprecationWarning = YES;
            }
            return override.get();
        }
    }

    auto osVersion = userAgentOSVersion.isEmpty() ? systemMarketingVersionForUserAgentString() : userAgentOSVersion;
    return makeString("Mozilla/5.0 ("_s, deviceNameForUserAgent(), "; CPU "_s, osNameForUserAgent(), ' ', osVersion, " like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko)"_s, separator, applicationName);
#endif
}

} // namespace WebCore.

#endif // PLATFORM(IOS_FAMILY)
