/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
#import "MediaPermissionUtilities.h"

#import "SandboxUtilities.h"
#import "WKWebViewInternal.h"
#import "WebPageProxy.h"
#import <WebCore/LocalizedStrings.h>
#import <WebCore/SecurityOriginData.h>
#import <mutex>
#import <wtf/BlockPtr.h>
#import <wtf/URLHelpers.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/spi/cf/CFBundleSPI.h>
#import <wtf/spi/darwin/SandboxSPI.h>

#if PLATFORM(IOS_FAMILY)
#import "UIKitUtilities.h"
#endif

#import "TCCSoftLink.h"
#import <pal/cocoa/AVFoundationSoftLink.h>
#import <pal/cocoa/SpeechSoftLink.h>

namespace WebKit {

bool checkSandboxRequirementForType(MediaPermissionType type)
{
#if PLATFORM(MAC)
    static std::once_flag audioFlag;
    static std::once_flag videoFlag;
    static bool isAudioEntitled = true;
    static bool isVideoEntitled = true;
    
    auto checkFunction = [](ASCIILiteral operation, bool* entitled) {
        if (!currentProcessIsSandboxed())
            return;

        int result = sandbox_check(getpid(), operation, static_cast<enum sandbox_filter_type>(SANDBOX_CHECK_NO_REPORT | SANDBOX_FILTER_NONE));
        if (result == -1)
            WTFLogAlways("Error checking '%s' sandbox access, errno=%ld", operation, (long)errno);
        *entitled = !result;
    };

    switch (type) {
    case MediaPermissionType::Audio:
        std::call_once(audioFlag, checkFunction, "device-microphone"_s, &isAudioEntitled);
        return isAudioEntitled;
    case MediaPermissionType::Video:
        std::call_once(videoFlag, checkFunction, "device-camera"_s, &isVideoEntitled);
        return isVideoEntitled;
    }
#endif
    return true;
}

bool checkUsageDescriptionStringForType(MediaPermissionType type)
{
    static std::once_flag audioDescriptionFlag;
    static std::once_flag videoDescriptionFlag;
    static bool hasMicrophoneDescriptionString = false;
    static bool hasCameraDescriptionString = false;

    switch (type) {
    case MediaPermissionType::Audio:
        static TCCAccessPreflightResult audioAccess = TCCAccessPreflight(get_TCC_kTCCServiceMicrophone(), NULL);
        if (audioAccess == kTCCAccessPreflightGranted)
            return true;
        std::call_once(audioDescriptionFlag, [] {
            hasMicrophoneDescriptionString = dynamic_objc_cast<NSString>(NSBundle.mainBundle.infoDictionary[@"NSMicrophoneUsageDescription"]).length > 0;
        });
        return hasMicrophoneDescriptionString;
    case MediaPermissionType::Video:
        static TCCAccessPreflightResult videoAccess = TCCAccessPreflight(get_TCC_kTCCServiceCamera(), NULL);
        if (videoAccess == kTCCAccessPreflightGranted)
            return true;
        std::call_once(videoDescriptionFlag, [] {
            hasCameraDescriptionString = dynamic_objc_cast<NSString>(NSBundle.mainBundle.infoDictionary[@"NSCameraUsageDescription"]).length > 0;
        });
        return hasCameraDescriptionString;
    }
}

bool checkUsageDescriptionStringForSpeechRecognition()
{
    return dynamic_objc_cast<NSString>(NSBundle.mainBundle.infoDictionary[@"NSSpeechRecognitionUsageDescription"]).length > 0;
}

static NSString* visibleDomain(const String& host)
{
    auto domain = WTF::URLHelpers::userVisibleURL(host.utf8());
    return startsWithLettersIgnoringASCIICase(domain, "www."_s) ? StringView(domain).substring(4).createNSString().autorelease() : static_cast<NSString *>(domain);
}

NSString *applicationVisibleNameFromOrigin(const WebCore::SecurityOriginData& origin)
{
    if (origin.protocol() != "http"_s && origin.protocol() != "https"_s)
        return nil;

    return visibleDomain(origin.host());
}

NSString *applicationVisibleName()
{
    NSBundle *appBundle = [NSBundle mainBundle];
    NSString *displayName = appBundle.infoDictionary[(__bridge NSString *)_kCFBundleDisplayNameKey];
    NSString *readableName = appBundle.infoDictionary[(__bridge NSString *)kCFBundleNameKey];
    return displayName ?: readableName;
}

static NSString *alertMessageText(MediaPermissionReason reason, const WebCore::SecurityOriginData& origin)
{
    NSString *visibleOrigin = applicationVisibleNameFromOrigin(origin);
    if (!visibleOrigin)
        visibleOrigin = applicationVisibleName();

    switch (reason) {
    case MediaPermissionReason::Camera:
        return [NSString stringWithFormat:WEB_UI_NSSTRING(@"Allow â€œ%@â€ to use your camera?", @"Message for user camera access prompt"), visibleOrigin];
    case MediaPermissionReason::CameraAndMicrophone:
        return [NSString stringWithFormat:WEB_UI_NSSTRING(@"Allow â€œ%@â€ to use your camera and microphone?", @"Message for user media prompt"), visibleOrigin];
    case MediaPermissionReason::Microphone:
        return [NSString stringWithFormat:WEB_UI_NSSTRING(@"Allow â€œ%@â€ to use your microphone?", @"Message for user microphone access prompt"), visibleOrigin];
    case MediaPermissionReason::ScreenCapture:
        return [NSString stringWithFormat:WEB_UI_NSSTRING(@"Allow â€œ%@â€ to observe your screen?", @"Message for screen sharing prompt"), visibleOrigin];
    case MediaPermissionReason::DeviceOrientation:
        return [NSString stringWithFormat:WEB_UI_NSSTRING(@"â€œ%@â€ Would Like to Access Motion and Orientation", @"Message for requesting access to the device motion and orientation"), visibleOrigin];
    case MediaPermissionReason::Geolocation:
        return [NSString stringWithFormat:WEB_UI_NSSTRING(@"Allow â€œ%@â€ to use your current location?", @"Message for geolocation prompt"), visibleOrigin];
    case MediaPermissionReason::SpeechRecognition:
        return [NSString stringWithFormat:WEB_UI_NSSTRING(@"Allow â€œ%@â€ to capture your audio and use it for speech recognition?", @"Message for spechrecognition prompt"), visibleDomain(origin.host())];
    }
}

static NSString *allowButtonText(MediaPermissionReason reason)
{
    switch (reason) {
    case MediaPermissionReason::Camera:
    case MediaPermissionReason::CameraAndMicrophone:
    case MediaPermissionReason::Microphone:
        return WEB_UI_STRING_KEY(@"Allow", "Allow (usermedia)", @"Allow button title in user media prompt");
    case MediaPermissionReason::ScreenCapture:
        return WEB_UI_STRING_KEY(@"Allow", "Allow (screensharing)", @"Allow button title in screen sharing prompt");
    case MediaPermissionReason::DeviceOrientation:
        return WEB_UI_STRING_KEY(@"Allow", "Allow (device motion and orientation access)", @"Button title in Device Orientation Permission API prompt");
    case MediaPermissionReason::Geolocation:
        return WEB_UI_STRING_KEY(@"Allow", "Allow (geolocation)", @"Allow button title in geolocation prompt");
    case MediaPermissionReason::SpeechRecognition:
        return WEB_UI_STRING_KEY(@"Allow", "Allow (speechrecognition)", @"Allow button title in speech recognition prompt");
    }
}

static NSString *doNotAllowButtonText(MediaPermissionReason reason)
{
    switch (reason) {
    case MediaPermissionReason::Camera:
    case MediaPermissionReason::CameraAndMicrophone:
    case MediaPermissionReason::Microphone:
        return WEB_UI_STRING_KEY(@"Donâ€™t Allow", "Donâ€™t Allow (usermedia)", @"Disallow button title in user media prompt");
    case MediaPermissionReason::ScreenCapture:
        return WEB_UI_STRING_KEY(@"Donâ€™t Allow", "Donâ€™t Allow (screensharing)", @"Disallow button title in screen sharing prompt");
    case MediaPermissionReason::DeviceOrientation:
        return WEB_UI_STRING_KEY(@"Cancel", "Cancel (device motion and orientation access)", @"Button title in Device Orientation Permission API prompt");
    case MediaPermissionReason::Geolocation:
        return WEB_UI_STRING_KEY(@"Donâ€™t Allow", "Donâ€™t Allow (geolocation)", @"Disallow button title in geolocation prompt");
    case MediaPermissionReason::SpeechRecognition:
        return WEB_UI_STRING_KEY(@"Donâ€™t Allow", "Donâ€™t Allow (speechrecognition)", @"Disallow button title in speech recognition prompt");
    }
}

void alertForPermission(WebPageProxy& page, MediaPermissionReason reason, const WebCore::SecurityOriginData& origin, CompletionHandler<void(bool)>&& completionHandler)
{
    ASSERT(isMainRunLoop());

#if PLATFORM(IOS_FAMILY)
    if (reason == MediaPermissionReason::DeviceOrientation) {
        if (auto& userPermissionHandler = page.deviceOrientationUserPermissionHandlerForTesting())
            return completionHandler(userPermissionHandler());
    }
#endif

    auto webView = page.cocoaView();
    if (!webView) {
        completionHandler(false);
        return;
    }
    
    auto *alertTitle = alertMessageText(reason, origin);
    if (!alertTitle) {
        completionHandler(false);
        return;
    }

    auto *allowButtonString = allowButtonText(reason);
    auto *doNotAllowButtonString = doNotAllowButtonText(reason);
    auto completionBlock = makeBlockPtr(WTFMove(completionHandler));

#if PLATFORM(MAC)
    auto alert = adoptNS([NSAlert new]);
    [alert setMessageText:alertTitle];
    NSButton *button = [alert addButtonWithTitle:allowButtonString];
    button.keyEquivalent = @"";
    button = [alert addButtonWithTitle:doNotAllowButtonString];
    button.keyEquivalent = @"\E";
    [alert beginSheetModalForWindow:[webView window] completionHandler:[completionBlock](NSModalResponse returnCode) {
        auto shouldAllow = returnCode == NSAlertFirstButtonReturn;
        completionBlock(shouldAllow);
    }];
#else
    auto alert = WebKit::createUIAlertController(alertTitle, nil);
    UIAlertAction* allowAction = [UIAlertAction actionWithTitle:allowButtonString style:UIAlertActionStyleDefault handler:[completionBlock](UIAlertAction *action) {
        completionBlock(true);
    }];

    UIAlertAction* doNotAllowAction = [UIAlertAction actionWithTitle:doNotAllowButtonString style:UIAlertActionStyleCancel handler:[completionBlock](UIAlertAction *action) {
        completionBlock(false);
    }];

    [alert addAction:doNotAllowAction];
    [alert addAction:allowAction];

    [[webView _wk_viewControllerForFullScreenPresentation] presentViewController:alert.get() animated:YES completion:nil];
#endif
}



void requestAVCaptureAccessForType(MediaPermissionType type, CompletionHandler<void(bool authorized)>&& completionHandler)
{
    ASSERT(isMainRunLoop());

#if HAVE(AVCAPTUREDEVICE)
    AVMediaType mediaType = type == MediaPermissionType::Audio ? AVMediaTypeAudio : AVMediaTypeVideo;
    auto decisionHandler = makeBlockPtr([completionHandler = WTFMove(completionHandler)](BOOL authorized) mutable {
        callOnMainRunLoop([completionHandler = WTFMove(completionHandler), authorized]() mutable {
            completionHandler(authorized);
        });
    });
    [PAL::getAVCaptureDeviceClass() requestAccessForMediaType:mediaType completionHandler:decisionHandler.get()];
#else
    UNUSED_PARAM(type);
    completionHandler(false);
#endif
}

MediaPermissionResult checkAVCaptureAccessForType(MediaPermissionType type)
{
#if HAVE(AVCAPTUREDEVICE)
    AVMediaType mediaType = type == MediaPermissionType::Audio ? AVMediaTypeAudio : AVMediaTypeVideo;
    auto authorizationStatus = [PAL::getAVCaptureDeviceClass() authorizationStatusForMediaType:mediaType];
    if (authorizationStatus == AVAuthorizationStatusDenied || authorizationStatus == AVAuthorizationStatusRestricted)
        return MediaPermissionResult::Denied;
    if (authorizationStatus == AVAuthorizationStatusNotDetermined)
        return MediaPermissionResult::Unknown;
    return MediaPermissionResult::Granted;
#else
    UNUSED_PARAM(type);
    return MediaPermissionResult::Denied;
#endif
}

#if HAVE(SPEECHRECOGNIZER)

void requestSpeechRecognitionAccess(CompletionHandler<void(bool authorized)>&& completionHandler)
{
    ASSERT(isMainRunLoop());

    auto decisionHandler = makeBlockPtr([completionHandler = WTFMove(completionHandler)](SFSpeechRecognizerAuthorizationStatus status) mutable {
        bool authorized = status == SFSpeechRecognizerAuthorizationStatusAuthorized;
        callOnMainRunLoop([completionHandler = WTFMove(completionHandler), authorized]() mutable {
            completionHandler(authorized);
        });
    });
    [PAL::getSFSpeechRecognizerClass() requestAuthorization:decisionHandler.get()];
}

MediaPermissionResult checkSpeechRecognitionServiceAccess()
{
    auto authorizationStatus = [PAL::getSFSpeechRecognizerClass() authorizationStatus];
IGNORE_WARNINGS_BEGIN("deprecated-enum-compare")
    if (authorizationStatus == SFSpeechRecognizerAuthorizationStatusDenied || authorizationStatus == SFSpeechRecognizerAuthorizationStatusRestricted)
        return MediaPermissionResult::Denied;
    if (authorizationStatus == SFSpeechRecognizerAuthorizationStatusAuthorized)
        return MediaPermissionResult::Granted;
IGNORE_WARNINGS_END
    return MediaPermissionResult::Unknown;
}

bool checkSpeechRecognitionServiceAvailability(const String& localeIdentifier)
{
    auto recognizer = localeIdentifier.isEmpty() ? adoptNS([PAL::allocSFSpeechRecognizerInstance() init]) : adoptNS([PAL::allocSFSpeechRecognizerInstance() initWithLocale:[NSLocale localeWithLocaleIdentifier:localeIdentifier]]);
    return recognizer && [recognizer isAvailable];
}

#endif // HAVE(SPEECHRECOGNIZER)

} // namespace WebKit
