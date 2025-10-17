/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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
#import "WKWebGeolocationPolicyDecider.h"

#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import "UIKitUtilities.h"
#import "WKWebViewPrivateForTesting.h"
#import <CoreLocation/CoreLocation.h>
#import <WebCore/LocalizedStrings.h>
#import <WebCore/SecurityOrigin.h>
#import <pal/spi/cocoa/NSFileManagerSPI.h>
#import <wtf/Deque.h>
#import <wtf/SoftLinking.h>
#import <wtf/WeakObjCPtr.h>
#import <wtf/spi/cf/CFBundleSPI.h>

SOFT_LINK_FRAMEWORK(CoreLocation)

SOFT_LINK_CLASS(CoreLocation, CLLocationManager)

static const NSInteger kGeolocationChallengeThreshold = 2;
static constexpr Seconds kGeolocationChallengeTimeout = 24_h;
static NSString * const kGeolocationChallengeCount = @"ChallengeCount";
static NSString * const kGeolocationChallengeDate = @"ChallengeDate";
static CFStringRef CLAppResetChangedNotification = CFSTR("com.apple.locationd.appreset");

static void clearGeolocationCache(CFNotificationCenterRef center, void *observer, CFStringRef name, const void *object, CFDictionaryRef userInfo)
{
    [(WKWebGeolocationPolicyDecider *)observer clearCache];
}

static bool appHasPreciseLocationPermission()
{
    auto locationManager = adoptNS([allocCLLocationManagerInstance() init]);

    CLAuthorizationStatus authStatus = [locationManager authorizationStatus];
    return (authStatus == kCLAuthorizationStatusAuthorizedAlways || authStatus == kCLAuthorizationStatusAuthorizedWhenInUse)
        && [locationManager accuracyAuthorization] == CLAccuracyAuthorizationFullAccuracy;
}

static NSString *appDisplayName()
{
    auto *bundle = [NSBundle mainBundle];
    NSString *displayName = [bundle objectForInfoDictionaryKey:(__bridge NSString *)_kCFBundleDisplayNameKey];
    if (!displayName)
        displayName = [bundle objectForInfoDictionaryKey:(__bridge NSString *)kCFBundleNameKey];
    if (!displayName)
        displayName = [bundle objectForInfoDictionaryKey:(__bridge NSString *)kCFBundleExecutableKey];
    if (!displayName)
        displayName = [bundle bundleIdentifier];
    return displayName;
}

static NSString *getToken(const WebCore::SecurityOriginData& securityOrigin, NSURL *requestingURL)
{
    if ([requestingURL isFileURL])
        return [requestingURL path];
    return securityOrigin.host();
}

struct PermissionRequest {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    static std::unique_ptr<PermissionRequest> create(const WebCore::SecurityOriginData& origin, NSURL *requestingURL, WKWebView *view, id<WKWebAllowDenyPolicyListener> listener)
    {
        auto request = std::unique_ptr<PermissionRequest>(new PermissionRequest);
        request->token = getToken(origin, requestingURL);
        request->requestingURL = requestingURL;
        request->view = view;
        request->listener = listener;
        return request;
    }

    RetainPtr<NSString> token;
    RetainPtr<NSString> domain;
    RetainPtr<NSURL> requestingURL;
    RetainPtr<WKWebView> view;
    RetainPtr<id<WKWebAllowDenyPolicyListener>> listener;
};

@implementation WKWebGeolocationPolicyDecider {
@private
    RetainPtr<dispatch_queue_t> _diskDispatchQueue;
    RetainPtr<NSMutableDictionary> _sites;
    Deque<std::unique_ptr<PermissionRequest>> _challenges;
    std::unique_ptr<PermissionRequest> _activeChallenge;
}

+ (instancetype)sharedPolicyDecider
{
    static WKWebGeolocationPolicyDecider *policyDecider = nil;
    if (!policyDecider)
        policyDecider = [[WKWebGeolocationPolicyDecider alloc] init];
    return policyDecider;
}

- (id)init
{
    self = [super init];
    if (!self)
        return nil;

    _diskDispatchQueue = adoptNS(dispatch_queue_create("com.apple.WebKit.WKWebGeolocationPolicyDecider", DISPATCH_QUEUE_SERIAL));

    CFNotificationCenterAddObserver(CFNotificationCenterGetDarwinNotifyCenter(), self, clearGeolocationCache, CLAppResetChangedNotification, NULL, CFNotificationSuspensionBehaviorCoalesce);

    return self;
}

- (void)dealloc
{
    CFNotificationCenterRemoveObserver(CFNotificationCenterGetDarwinNotifyCenter(), self, CLAppResetChangedNotification, NULL);
    [super dealloc];
}

- (void)decidePolicyForGeolocationRequestFromOrigin:(const WebCore::SecurityOriginData&)securityOrigin requestingURL:(NSURL *)requestingURL view:(WKWebView *)view listener:(id<WKWebAllowDenyPolicyListener>)listener
{
    auto permissionRequest = PermissionRequest::create(securityOrigin, requestingURL, view, listener);
    _challenges.append(WTFMove(permissionRequest));
    [self _executeNextChallenge];
}

- (void)_executeNextChallenge
{
    if (_challenges.isEmpty())
        return;
    if (_activeChallenge)
        return;

    _activeChallenge = _challenges.takeFirst();
    [self _loadWithCompletionHandler:^{
        if ([_activeChallenge->view _shouldBypassGeolocationPromptForTesting]) {
            [self _finishActiveChallenge:YES];
            return;
        }
        NSInteger challengeCount = [self _getChallengeCountFromHistoryForToken:_activeChallenge->token.get() requestingURL:_activeChallenge->requestingURL.get()];
        if (challengeCount >= kGeolocationChallengeThreshold) {
            [self _finishActiveChallenge:YES];
            return;
        }
        if (challengeCount <= -kGeolocationChallengeThreshold) {
            [self _finishActiveChallenge:NO];
            return;
        }

        NSString *applicationName = appDisplayName();
        NSString *message;

    IGNORE_WARNINGS_BEGIN("format-nonliteral")
        NSString *title = [NSString stringWithFormat:WEB_UI_STRING("â€œ%@â€ would like to use your current location.", "Prompt for a webpage to request location access. The parameter is the domain for the webpage."), _activeChallenge->token.get()];
        if (appHasPreciseLocationPermission())
            message = [NSString stringWithFormat:WEB_UI_STRING("This website will use your precise location because â€œ%@â€ currently has access to your precise location.", "Message informing the user that the website will have precise location data"), applicationName];
        else
            message = [NSString stringWithFormat:WEB_UI_STRING("This website will use your approximate location because â€œ%@â€ currently has access to your approximate location.", "Message informing the user that the website will have approximate location data"), applicationName];
    IGNORE_WARNINGS_END

        NSString *allowActionTitle = WEB_UI_STRING("Allow", "Action authorizing a webpage to access the userâ€™s location.");
        NSString *denyActionTitle = WEB_UI_STRING_KEY("Donâ€™t Allow", "Donâ€™t Allow (website location dialog)", "Action denying a webpage access to the userâ€™s location.");

        auto alert = WebKit::createUIAlertController(title, message);
        UIAlertAction *denyAction = [UIAlertAction actionWithTitle:denyActionTitle style:UIAlertActionStyleDefault handler:[weakSelf = WeakObjCPtr<WKWebGeolocationPolicyDecider>(self)](UIAlertAction *) mutable {
            if (auto strongSelf = weakSelf.get())
                [strongSelf _finishActiveChallenge:NO];
        }];
        UIAlertAction *allowAction = [UIAlertAction actionWithTitle:allowActionTitle style:UIAlertActionStyleDefault handler:[weakSelf = WeakObjCPtr<WKWebGeolocationPolicyDecider>(self)](UIAlertAction *) mutable {
            if (auto strongSelf = weakSelf.get())
                [strongSelf _finishActiveChallenge:YES];
        }];

        [alert addAction:denyAction];
        [alert addAction:allowAction];

        [[_activeChallenge->view _wk_viewControllerForFullScreenPresentation] presentViewController:alert.get() animated:YES completion:nil];
    }];
}

- (void)_finishActiveChallenge:(BOOL)allow
{
    if (!_activeChallenge)
        return;

    [self _addChallengeCount:allow ? 1 : -1 forToken:_activeChallenge->token.get() requestingURL:_activeChallenge->requestingURL.get()];
    if (allow)
        [_activeChallenge->listener allow];
    else
        [_activeChallenge->listener deny];
    _activeChallenge = nullptr;
    [self _executeNextChallenge];
}

- (void)clearCache
{
    _sites = nil;

    dispatch_async(_diskDispatchQueue.get(), ^{
        [[NSFileManager defaultManager] _web_removeFileOnlyAtPath:[self _siteFile]];
    });
}

- (NSString *)_siteFileInContainerDirectory:(NSString *)containerDirectory creatingIntermediateDirectoriesIfNecessary:(BOOL)createIntermediateDirectories
{
    NSString *webKitDirectory = [containerDirectory stringByAppendingPathComponent:@"Library/WebKit"];
    if (createIntermediateDirectories)
        [[NSFileManager defaultManager] _web_createDirectoryAtPathWithIntermediateDirectories:webKitDirectory attributes:nil];
    return [webKitDirectory stringByAppendingPathComponent:@"GeolocationSitesV2.plist"];
}

- (NSString *)_siteFile
{
    static NSString *sSiteFile = nil;
    if (!sSiteFile)
        sSiteFile = [[self _siteFileInContainerDirectory:NSHomeDirectory() creatingIntermediateDirectoriesIfNecessary:YES] retain];
    return sSiteFile;
}

static RetainPtr<NSMutableDictionary> createChallengeDictionary(NSData *data)
{
    NSError *error = nil;
    NSPropertyListFormat dbFormat = NSPropertyListBinaryFormat_v1_0;
    return [NSPropertyListSerialization propertyListWithData:data options:NSPropertyListMutableContainersAndLeaves format:&dbFormat error:&error];
}

- (void)_loadWithCompletionHandler:(void (^)())completionHandler
{
    ASSERT(isMainRunLoop());
    if (_sites) {
        completionHandler();
        return;
    }

    dispatch_async(_diskDispatchQueue.get(), ^{
        RetainPtr<NSMutableDictionary> sites;
        RetainPtr<NSData> data = [NSData dataWithContentsOfFile:[self _siteFile] options:NSDataReadingMappedIfSafe error:NULL];
        if (data)
            sites = createChallengeDictionary(data.get());
        else
            sites = adoptNS([[NSMutableDictionary alloc] init]);

        dispatch_async(dispatch_get_main_queue(), ^{
            if (!_sites)
                _sites = sites;
            completionHandler();
        });
    });
}

- (void)_save
{
    if (![_sites count])
        return;

    NSError *error = nil;
    NSData *data = [NSPropertyListSerialization dataWithPropertyList:_sites.get() format:NSPropertyListBinaryFormat_v1_0 options:0 error:&error];
    if (!data)
        return;

    NSString *siteFilePath = [self _siteFile];
    dispatch_async(_diskDispatchQueue.get(), ^{
        [data writeToFile:siteFilePath atomically:YES];
    });
}

- (NSInteger)_getChallengeCountFromHistoryForToken:(NSString *)token requestingURL:(NSURL *)requestingURL
{
    NSDictionary *challengeHistory = (NSDictionary *)[_sites objectForKey:token];
    if (challengeHistory && ![[requestingURL scheme] isEqualToString:@"data"]) {
        NSDate *lastChallengeDate = [challengeHistory objectForKey:kGeolocationChallengeDate];
        NSDate *expirationDate = [lastChallengeDate dateByAddingTimeInterval:kGeolocationChallengeTimeout.seconds()];
        if ([expirationDate compare:[NSDate date]] != NSOrderedAscending)
            return [(NSNumber *)[challengeHistory objectForKey:kGeolocationChallengeCount] integerValue];
    }
    return 0;
}

- (void)_addChallengeCount:(NSInteger)count forToken:(NSString *)token requestingURL:(NSURL *)requestingURL
{
    NSInteger challengeCount = [self _getChallengeCountFromHistoryForToken:token requestingURL:requestingURL];
    challengeCount += count;

    NSDictionary *savedChallenge = @{
        kGeolocationChallengeCount : @(challengeCount),
        kGeolocationChallengeDate : [NSDate date]
    };
    [_sites setObject:savedChallenge forKey:token];
    [self _save];
}

@end

#endif // PLATFORM(IOS_FAMILY)
