/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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
#import "WebPreferencesDefaultValuesIOS.h"

#if PLATFORM(IOS_FAMILY)

#import <pal/spi/cocoa/FeatureFlagsSPI.h>
#import <pal/spi/ios/ManagedConfigurationSPI.h>
#import <pal/system/ios/Device.h>
#import <pal/system/ios/UserInterfaceIdiom.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>

#import <pal/ios/ManagedConfigurationSoftLink.h>

namespace WebKit {

#if ENABLE(TEXT_AUTOSIZING)

bool defaultTextAutosizingUsesIdempotentMode()
{
    return !PAL::currentUserInterfaceIdiomIsSmallScreen();
}

#endif

#if !PLATFORM(MACCATALYST) && !PLATFORM(WATCHOS)
static std::optional<bool>& cachedAllowsRequest()
{
    static NeverDestroyed<std::optional<bool>> allowsRequest;
    return allowsRequest;
}

bool allowsDeprecatedSynchronousXMLHttpRequestDuringUnload()
{
    if (!cachedAllowsRequest())
        cachedAllowsRequest() = [(MCProfileConnection *)[PAL::getMCProfileConnectionClass() sharedConnection] effectiveBoolValueForSetting:@"allowDeprecatedWebKitSynchronousXHRLoads"] == MCRestrictedBoolExplicitYes;
    return *cachedAllowsRequest();
}

void setAllowsDeprecatedSynchronousXMLHttpRequestDuringUnload(bool allowsRequest)
{
    cachedAllowsRequest() = allowsRequest;
}
#endif

#if ENABLE(MEDIA_SOURCE)

bool defaultMediaSourceEnabled()
{
#if PLATFORM(APPLETV)
    return true;
#else
    return !PAL::deviceClassIsSmallScreen();
#endif
}

#endif

static bool isAsyncTextInputFeatureFlagEnabled()
{
    static bool enabled = false;
#if USE(BROWSERENGINEKIT)
    static std::once_flag flag;
    std::call_once(flag, [] {
        if (PAL::deviceClassIsSmallScreen())
            enabled = os_feature_enabled(UIKit, async_text_input_iphone) || os_feature_enabled(UIKit, async_text_input);
        else if (PAL::deviceHasIPadCapability())
            enabled = os_feature_enabled(UIKit, async_text_input_ipad);
    });
#endif
    return enabled;
}

bool defaultUseAsyncUIKitInteractions()
{
    if (WTF::CocoaApplication::isIBooks()) {
        // FIXME: Remove this exception once rdar://119836700 is addressed.
        return false;
    }
    return isAsyncTextInputFeatureFlagEnabled();
}

bool defaultWriteRichTextDataWhenCopyingOrDragging()
{
    // While this is keyed off of the same underlying system feature flag as
    // "Async UIKit Interactions", the logic is inverted, since versions of
    // iOS with the requisite support for async text input *no longer* require
    // WebKit to write RTF and attributed string data.
    return !isAsyncTextInputFeatureFlagEnabled();
}

bool defaultAutomaticLiveResizeEnabled()
{
#if PLATFORM(VISION)
    return true;
#elif USE(BROWSERENGINEKIT)
    static bool enabled = PAL::deviceHasIPadCapability() && os_feature_enabled(UIKit, async_text_input_ipad);
    return enabled;
#else
    return false;
#endif
}

#if USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/WebPreferencesDefaultValuesIOSAdditions.mm>)
#import <WebKitAdditions/WebPreferencesDefaultValuesIOSAdditions.mm>
#else
bool defaultVisuallyContiguousBidiTextSelectionEnabled()
{
    return false;
}
bool defaultBidiContentAwarePasteEnabled()
{
    return false;
}
#endif

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
