/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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
#import "DefaultWebBrowserChecks.h"

#import "AuxiliaryProcess.h"
#import "Connection.h"
#import "Logging.h"
#import <WebCore/RegistrableDomain.h>
#import <wtf/HashMap.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/RunLoop.h>
#import <wtf/RuntimeApplicationChecks.h>
#import <wtf/WorkQueue.h>
#import <wtf/cocoa/Entitlements.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#import <wtf/text/StringHash.h>

#import "TCCSoftLink.h"

namespace WebKit {

static bool isFullWebBrowserOrRunningTest(const String&);

static bool treatAsNonBrowser(const String& bundleID)
{
    return bundleID == "inAppBrowserPrivacyTestIdentifier"_s;
}

bool isRunningTest(const String& bundleID)
{
    return bundleID == "com.apple.WebKit.TestWebKitAPI"_s || bundleID == "com.apple.WebKit.WebKitTestRunner"_s || bundleID == "org.webkit.WebKitTestRunnerApp"_s;
}

std::span<const WebCore::RegistrableDomain> appBoundDomainsForTesting(const String& bundleID)
{
    if (bundleID == "inAppBrowserPrivacyTestIdentifier"_s) {
        static NeverDestroyed domains = std::array {
            WebCore::RegistrableDomain::uncheckedCreateFromRegistrableDomainString("127.0.0.1"_s),
        };
        return domains.get();
    }
    return { };
}

#if ASSERT_ENABLED
static bool isInWebKitChildProcess()
{
    static bool isInSubProcess;

    static dispatch_once_t once;
    dispatch_once(&once, ^{
#if USE(EXTENSIONKIT)
        isInSubProcess |= WTF::processHasEntitlement("com.apple.developer.web-browser-engine.networking"_s)
            || WTF::processHasEntitlement("com.apple.developer.web-browser-engine.rendering"_s)
            || WTF::processHasEntitlement("com.apple.developer.web-browser-engine.webcontent"_s);
        if (isInSubProcess)
            return;
#endif // USE(EXTENSIONKIT)
        NSString *bundleIdentifier = [[NSBundle mainBundle] bundleIdentifier];
        isInSubProcess = [bundleIdentifier hasPrefix:@"com.apple.WebKit.WebContent"]
            || [bundleIdentifier hasPrefix:@"com.apple.WebKit.Networking"]
            || [bundleIdentifier hasPrefix:@"com.apple.WebKit.GPU"];
#if ENABLE(MODEL_PROCESS)
        isInSubProcess = isInSubProcess || [bundleIdentifier hasPrefix:@"com.apple.WebKit.Model"];
#endif // ENABLE(MODEL_PROCESS)
    });

    return isInSubProcess;
}
#endif

enum class TrackingPreventionState : uint8_t {
    Uninitialized,
    Enabled,
    Disabled
};

static std::atomic<TrackingPreventionState> currentTrackingPreventionState = TrackingPreventionState::Uninitialized;

bool hasRequestedCrossWebsiteTrackingPermission()
{
    ASSERT(!isInWebKitChildProcess());

    static std::atomic<bool> hasRequestedCrossWebsiteTrackingPermission = [[NSBundle mainBundle] objectForInfoDictionaryKey:@"NSCrossWebsiteTrackingUsageDescription"];
    return hasRequestedCrossWebsiteTrackingPermission;
}

static bool determineTrackingPreventionStateInternal(bool appWasLinkedOnOrAfter, const String& bundleIdentifier)
{
    ASSERT(!RunLoop::isMain());
    ASSERT(!isInWebKitChildProcess());

#if ENABLE(APP_BOUND_DOMAINS)
    bool isFullWebBrowser = isFullWebBrowserOrRunningTest(bundleIdentifier);
#else
    bool isFullWebBrowser = isRunningTest(bundleIdentifier);
#endif
    if (!appWasLinkedOnOrAfter && !isFullWebBrowser)
        return false;

    if (!isFullWebBrowser && !hasRequestedCrossWebsiteTrackingPermission())
        return true;

    TCCAccessPreflightResult result = kTCCAccessPreflightDenied;
#if PLATFORM(IOS) || PLATFORM(MAC) || PLATFORM(VISION)
    result = TCCAccessPreflight(get_TCC_kTCCServiceWebKitIntelligentTrackingPrevention(), nullptr);
#endif
    return result != kTCCAccessPreflightDenied;
}

static RefPtr<WorkQueue>& itpQueue()
{
    static NeverDestroyed<RefPtr<WorkQueue>> itpQueue;
    return itpQueue;
}

void determineTrackingPreventionState()
{
    ASSERT(RunLoop::isMain());
    if (currentTrackingPreventionState != TrackingPreventionState::Uninitialized)
        return;

    bool appWasLinkedOnOrAfter = linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::SessionCleanupByDefault);

    itpQueue() = WorkQueue::create("com.apple.WebKit.itpCheckQueue"_s);
    itpQueue()->dispatch([appWasLinkedOnOrAfter, bundleIdentifier = applicationBundleIdentifier().isolatedCopy()] {
        currentTrackingPreventionState = determineTrackingPreventionStateInternal(appWasLinkedOnOrAfter, bundleIdentifier) ? TrackingPreventionState::Enabled : TrackingPreventionState::Disabled;
        RunLoop::main().dispatch([] {
            itpQueue() = nullptr;
        });
    });
}

bool doesAppHaveTrackingPreventionEnabled()
{
    ASSERT(!isInWebKitChildProcess());
    ASSERT(RunLoop::isMain());
    // If we're still computing the ITP state on the background thread, then synchronize with it.
    if (itpQueue())
        itpQueue()->dispatchSync([] { });
    ASSERT(currentTrackingPreventionState != TrackingPreventionState::Uninitialized);
    return currentTrackingPreventionState == TrackingPreventionState::Enabled;
}

bool doesParentProcessHaveTrackingPreventionEnabled(AuxiliaryProcess& auxiliaryProcess, bool hasRequestedCrossWebsiteTrackingPermission)
{
    ASSERT(isInWebKitChildProcess());
    ASSERT(RunLoop::isMain());

    if (!isParentProcessAFullWebBrowser(auxiliaryProcess) && !hasRequestedCrossWebsiteTrackingPermission)
        return true;

    static bool trackingPreventionEnabled { true };
    static dispatch_once_t once;
    dispatch_once(&once, ^{

        TCCAccessPreflightResult result = kTCCAccessPreflightDenied;
#if PLATFORM(IOS) || PLATFORM(MAC) || PLATFORM(VISION)
        RefPtr<IPC::Connection> connection = auxiliaryProcess.parentProcessConnection();
        if (!connection) {
            ASSERT_NOT_REACHED();
            RELEASE_LOG_ERROR(IPC, "Unable to get parent process connection");
            return;
        }

        auto auditToken = connection->getAuditToken();
        if (!auditToken) {
            ASSERT_NOT_REACHED();
            RELEASE_LOG_ERROR(IPC, "Unable to get parent process audit token");
            return;
        }
        result = TCCAccessPreflightWithAuditToken(get_TCC_kTCCServiceWebKitIntelligentTrackingPrevention(), auditToken.value(), nullptr);
#endif
        trackingPreventionEnabled = result != kTCCAccessPreflightDenied;
    });
    return trackingPreventionEnabled;
}

static std::atomic<bool> hasCheckedUsageStrings = false;
bool hasProhibitedUsageStrings()
{
    ASSERT(!isInWebKitChildProcess());

    static bool hasProhibitedUsageStrings = false;

    if (hasCheckedUsageStrings)
        return hasProhibitedUsageStrings;

    NSDictionary *infoDictionary = [[NSBundle mainBundle] infoDictionary];
    RELEASE_ASSERT(infoDictionary);

    // See <rdar://problem/59979468> for details about how this list was selected.
    auto prohibitedStrings = @[
        @"NSHomeKitUsageDescription",
        @"NSBluetoothAlwaysUsageDescription",
        @"NSPhotoLibraryUsageDescription",
        @"NSHealthShareUsageDescription",
        @"NSHealthUpdateUsageDescription",
        @"NSLocationAlwaysUsageDescription",
        @"NSLocationAlwaysAndWhenInUseUsageDescription"
    ];

    for (NSString *prohibitedString : prohibitedStrings) {
        if ([infoDictionary objectForKey:prohibitedString]) {
            String message = [NSString stringWithFormat:@"[In-App Browser Privacy] %@ used prohibited usage string %@.", [[NSBundle mainBundle] bundleIdentifier], prohibitedString];
            WTFLogAlways(message.utf8().data());
            hasProhibitedUsageStrings = true;
            break;
        }
    }
    hasCheckedUsageStrings = true;
    return hasProhibitedUsageStrings;
}

bool isParentProcessAFullWebBrowser(AuxiliaryProcess& auxiliaryProcess)
{
    ASSERT(isInWebKitChildProcess());

    static bool fullWebBrowser { false };
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        RefPtr<IPC::Connection> connection = auxiliaryProcess.parentProcessConnection();
        if (!connection) {
            ASSERT_NOT_REACHED();
            RELEASE_LOG_ERROR(IPC, "Unable to get parent process connection");
            return;
        }

        auto auditToken = connection->getAuditToken();
        if (!auditToken) {
            ASSERT_NOT_REACHED();
            RELEASE_LOG_ERROR(IPC, "Unable to get parent process audit token");
            return;
        }

        fullWebBrowser = WTF::hasEntitlement(*auditToken, "com.apple.developer.web-browser"_s);
    });

    auto bundleID = applicationBundleIdentifier();

    if (isRunningTest(bundleID))
        return true;

    return fullWebBrowser && !treatAsNonBrowser(bundleID);
}

bool isFullWebBrowserOrRunningTest(const String& bundleIdentifier)
{
    ASSERT(!isInWebKitChildProcess());

#if ENABLE(APP_BOUND_DOMAINS)
    static bool fullWebBrowser = WTF::processHasEntitlement("com.apple.developer.web-browser"_s);
#elif PLATFORM(MAC)
    static bool fullWebBrowser;
    static std::once_flag once;
    std::call_once(once, [] {
        NSURL *currentURL = [[NSBundle mainBundle] bundleURL];
        NSArray<NSURL *> *httpURLs = [[NSWorkspace sharedWorkspace] URLsForApplicationsToOpenURL:[NSURL URLWithString:@"http:"]];
        bool canOpenHTTP = [httpURLs containsObject:currentURL];
        NSArray<NSURL *> *httpsURLs = [[NSWorkspace sharedWorkspace] URLsForApplicationsToOpenURL:[NSURL URLWithString:@"https:"]];
        bool canOpenHTTPS = [httpsURLs containsObject:currentURL];
        fullWebBrowser = canOpenHTTPS && canOpenHTTP;
    });
#else
    ASSERT_NOT_REACHED();
    static bool fullWebBrowser = false;
#endif

    if (isRunningTest(bundleIdentifier))
        return true;

    return fullWebBrowser && !treatAsNonBrowser(bundleIdentifier);
}

bool shouldEvaluateJavaScriptWithoutTransientActivation()
{
    static bool staticShouldEvaluateJavaScriptWithoutTransientActivation = [] {
        if (linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::EvaluateJavaScriptWithoutTransientActivation))
            return true;

        return isFullWebBrowserOrRunningTest();
    }();

    return staticShouldEvaluateJavaScriptWithoutTransientActivation;
}

bool isFullWebBrowserOrRunningTest()
{
    ASSERT(!isInWebKitChildProcess());
    ASSERT(RunLoop::isMain());

    return isFullWebBrowserOrRunningTest(applicationBundleIdentifier());
}

} // namespace WebKit
