/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#import "WebPreferencesDefaultValues.h"

#import "WebKitVersionChecks.h"
#import <Foundation/NSBundle.h>
#import <mach-o/dyld.h>
#import <pal/spi/cf/CFUtilitiesSPI.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#import <wtf/spi/darwin/dyldSPI.h>
#import <wtf/text/WTFString.h>

#if PLATFORM(IOS_FAMILY)
#import <pal/system/ios/Device.h>
#endif

namespace WebKit {

#if PLATFORM(IOS_FAMILY)

bool defaultAllowsInlineMediaPlayback()
{
    return !PAL::deviceClassIsSmallScreen();
}

bool defaultAllowsInlineMediaPlaybackAfterFullscreen()
{
    return PAL::deviceClassIsSmallScreen();
}

bool defaultAllowsPictureInPictureMediaPlayback()
{
    static bool shouldAllowPictureInPictureMediaPlayback = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::PictureInPictureMediaPlayback);
    return shouldAllowPictureInPictureMediaPlayback;
}

bool defaultJavaScriptCanOpenWindowsAutomatically()
{
    static bool shouldAllowWindowOpenWithoutUserGesture = WTF::IOSApplication::isTheSecretSocietyHiddenMystery() && !linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::NoTheSecretSocietyHiddenMysteryWindowOpenQuirk);
    return shouldAllowWindowOpenWithoutUserGesture;
}

bool defaultInlineMediaPlaybackRequiresPlaysInlineAttribute()
{
    return PAL::deviceClassIsSmallScreen();
}

bool defaultPassiveTouchListenersAsDefaultOnDocument()
{
    static bool result = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::DefaultsToPassiveTouchListenersOnDocument);
    return result;
}

bool defaultShowModalDialogEnabled()
{
    static bool newSDK = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::NoShowModalDialog);
    return !newSDK;
}

bool defaultRequiresUserGestureToLoadVideo()
{
    static bool shouldRequireUserGestureToLoadVideo = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::RequiresUserGestureToLoadVideo);
    return shouldRequireUserGestureToLoadVideo;
}

bool defaultWebSQLEnabled()
{
    // For backward compatibility, keep WebSQL working until apps are rebuilt with the iOS 14 SDK.
    static bool webSQLEnabled = !linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::WebSQLDisabledByDefaultInLegacyWebKit);
    return webSQLEnabled;
}

bool defaultAllowContentSecurityPolicySourceStarToMatchAnyProtocol()
{
    static bool shouldAllowContentSecurityPolicySourceStarToMatchAnyProtocol = !WebKitLinkedOnOrAfter(WEBKIT_FIRST_VERSION_WITH_CONTENT_SECURITY_POLICY_SOURCE_STAR_PROTOCOL_RESTRICTION);
    return shouldAllowContentSecurityPolicySourceStarToMatchAnyProtocol;
}

#endif // PLATFORM(IOS_FAMILY)

#if PLATFORM(MAC)

bool defaultLoadDeferringEnabled()
{
    return !WTF::MacApplication::isAdobeInstaller();
}

bool defaultWindowFocusRestricted()
{
    return !WTF::MacApplication::isHRBlock();
}

bool defaultUsePreHTML5ParserQuirks()
{
    // Mail.app must continue to display HTML email that contains quirky markup.
    return WTF::MacApplication::isAppleMail();
}

bool defaultNeedsAdobeFrameReloadingQuirk()
{
    static bool needsQuirk = _CFAppVersionCheckLessThan(CFSTR("com.adobe.Acrobat"), -1, 9.0)
        || _CFAppVersionCheckLessThan(CFSTR("com.adobe.Acrobat.Pro"), -1, 9.0)
        || _CFAppVersionCheckLessThan(CFSTR("com.adobe.Reader"), -1, 9.0)
        || _CFAppVersionCheckLessThan(CFSTR("com.adobe.distiller"), -1, 9.0)
        || _CFAppVersionCheckLessThan(CFSTR("com.adobe.Contribute"), -1, 4.2)
        || _CFAppVersionCheckLessThan(CFSTR("com.adobe.dreamweaver-9.0"), -1, 9.1)
        || _CFAppVersionCheckLessThan(CFSTR("com.macromedia.fireworks"), -1, 9.1)
        || _CFAppVersionCheckLessThan(CFSTR("com.adobe.InCopy"), -1, 5.1)
        || _CFAppVersionCheckLessThan(CFSTR("com.adobe.InDesign"), -1, 5.1)
        || _CFAppVersionCheckLessThan(CFSTR("com.adobe.Soundbooth"), -1, 2);

    return needsQuirk;
}

bool defaultScrollAnimatorEnabled()
{
    static bool enabled = [[NSUserDefaults standardUserDefaults] boolForKey:@"NSScrollAnimationEnabled"];
    return enabled;
}

bool defaultTreatsAnyTextCSSLinkAsStylesheet()
{
    static bool needsQuirk = !WebKitLinkedOnOrAfter(WEBKIT_FIRST_VERSION_WITHOUT_LINK_ELEMENT_TEXT_CSS_QUIRK)
        && _CFAppVersionCheckLessThan(CFSTR("com.e-frontier.shade10"), -1, 10.6);
    return needsQuirk;
}

bool defaultNeedsFrameNameFallbackToIdQuirk()
{
    static bool needsQuirk = _CFAppVersionCheckLessThan(CFSTR("info.colloquy"), -1, 2.5);
    return needsQuirk;
}

bool defaultNeedsKeyboardEventDisambiguationQuirks()
{
    static bool needsQuirks = !WebKitLinkedOnOrAfter(WEBKIT_FIRST_VERSION_WITH_IE_COMPATIBLE_KEYBOARD_EVENT_DISPATCH) && !WTF::MacApplication::isSafari();
    return needsQuirks;
}

#endif // PLATFORM(MAC)

bool defaultAttachmentElementEnabled()
{
#if PLATFORM(IOS_FAMILY)
    return WTF::IOSApplication::isMobileMail();
#else
    return WTF::MacApplication::isAppleMail();
#endif
}

bool defaultShouldRestrictBaseURLSchemes()
{
    static bool shouldRestrictBaseURLSchemes = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::RestrictsBaseURLSchemes);
    return shouldRestrictBaseURLSchemes;
}

bool defaultAllowDisplayOfInsecureContent()
{
    static bool shouldAllowDisplayOfInsecureContent = !WebKitLinkedOnOrAfter(WEBKIT_FIRST_VERSION_WITH_INSECURE_CONTENT_BLOCKING);
    return shouldAllowDisplayOfInsecureContent;
}

bool defaultAllowRunningOfInsecureContent()
{
    static bool shouldAllowRunningOfInsecureContent = !WebKitLinkedOnOrAfter(WEBKIT_FIRST_VERSION_WITH_INSECURE_CONTENT_BLOCKING);
    return shouldAllowRunningOfInsecureContent;
}

bool defaultShouldConvertInvalidURLsToBlank()
{
    static bool shouldConvertInvalidURLsToBlank = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::ConvertsInvalidURLsToBlank);
    return shouldConvertInvalidURLsToBlank;
}

bool defaultPopoverAttributeEnabled()
{
    static bool newSDK = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::PopoverAttributeEnabled);
    return newSDK;
}

#if PLATFORM(MAC)

bool defaultPassiveWheelListenersAsDefaultOnDocument()
{
    static bool result = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::DefaultsToPassiveWheelListenersOnDocument);
    return result;
}

bool defaultWheelEventGesturesBecomeNonBlocking()
{
    static bool result = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::AllowsWheelEventGesturesToBecomeNonBlocking);
    return result;
}

#endif

#if ENABLE(MEDIA_SOURCE) && PLATFORM(IOS_FAMILY)

bool defaultMediaSourceEnabled()
{
    return !PAL::deviceClassIsSmallScreen();
}

#endif

} // namespace WebKit
