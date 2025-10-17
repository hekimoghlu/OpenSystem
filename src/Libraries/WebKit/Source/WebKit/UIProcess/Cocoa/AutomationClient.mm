/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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
#import "AutomationClient.h"

#if ENABLE(REMOTE_INSPECTOR)

#import "WKProcessPool.h"
#import "_WKAutomationDelegate.h"
#import "_WKAutomationSessionConfiguration.h"
#import <JavaScriptCore/RemoteInspector.h>
#import <wtf/RunLoop.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/spi/cf/CFBundleSPI.h>
#import <wtf/text/WTFString.h>

using namespace Inspector;

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AutomationClient);

AutomationClient::AutomationClient(WKProcessPool *processPool, id <_WKAutomationDelegate> delegate)
    : m_processPool(processPool)
    , m_delegate(delegate)
{
    m_delegateMethods.allowsRemoteAutomation = [delegate respondsToSelector:@selector(_processPoolAllowsRemoteAutomation:)];
    m_delegateMethods.requestAutomationSession = [delegate respondsToSelector:@selector(_processPool:didRequestAutomationSessionWithIdentifier:configuration:)];
    m_delegateMethods.requestedDebuggablesToWakeUp = [delegate respondsToSelector:@selector(_processPoolDidRequestInspectorDebuggablesToWakeUp:)];
    m_delegateMethods.browserNameForAutomation = [delegate respondsToSelector:@selector(_processPoolBrowserNameForAutomation:)];
    m_delegateMethods.browserVersionForAutomation = [delegate respondsToSelector:@selector(_processPoolBrowserVersionForAutomation:)];

    RemoteInspector::singleton().setClient(this);
}

AutomationClient::~AutomationClient()
{
    RemoteInspector::singleton().setClient(nullptr);
}

// MARK: API::AutomationClient

void AutomationClient::didRequestAutomationSession(WebKit::WebProcessPool*, const String& sessionIdentifier)
{
    requestAutomationSession(sessionIdentifier, { });
}

// MARK: RemoteInspector::Client

bool AutomationClient::remoteAutomationAllowed() const
{
    if (m_delegateMethods.allowsRemoteAutomation)
        return [m_delegate.get() _processPoolAllowsRemoteAutomation:m_processPool];

    return false;
}

void AutomationClient::requestAutomationSession(const String& sessionIdentifier, const RemoteInspector::Client::SessionCapabilities& sessionCapabilities)
{
    auto configuration = adoptNS([[_WKAutomationSessionConfiguration alloc] init]);
    [configuration setAcceptInsecureCertificates:sessionCapabilities.acceptInsecureCertificates];
    
    if (sessionCapabilities.allowInsecureMediaCapture)
        [configuration setAllowsInsecureMediaCapture:sessionCapabilities.allowInsecureMediaCapture.value()];
    if (sessionCapabilities.suppressICECandidateFiltering)
        [configuration setSuppressesICECandidateFiltering:sessionCapabilities.suppressICECandidateFiltering.value()];

    // Force clients to create and register a session asynchronously. Otherwise,
    // RemoteInspector will try to acquire its lock to register the new session and
    // deadlock because it's already taken while handling XPC messages.
    NSString *requestedSessionIdentifier = sessionIdentifier;
    RunLoop::main().dispatch([this, requestedSessionIdentifier = retainPtr(requestedSessionIdentifier), configuration = WTFMove(configuration)] {
        if (m_delegateMethods.requestAutomationSession)
            [m_delegate.get() _processPool:m_processPool didRequestAutomationSessionWithIdentifier:requestedSessionIdentifier.get() configuration:configuration.get()];
    });
}

// FIXME: Consider renaming AutomationClient and _WKAutomationDelegate to _WKInspectorDelegate since it isn't only used for automation now.
// http://webkit.org/b/221933
void AutomationClient::requestedDebuggablesToWakeUp()
{
    RunLoop::main().dispatch([this] {
        if (m_delegateMethods.requestedDebuggablesToWakeUp)
            [m_delegate.get() _processPoolDidRequestInspectorDebuggablesToWakeUp:m_processPool];
    });
}

String AutomationClient::browserName() const
{
    if (m_delegateMethods.browserNameForAutomation)
        return [m_delegate _processPoolBrowserNameForAutomation:m_processPool];

    // Fall back to using the unlocalized app name (i.e., 'Safari').
    NSBundle *appBundle = [NSBundle mainBundle];
    NSString *displayName = appBundle.infoDictionary[(__bridge NSString *)_kCFBundleDisplayNameKey];
    NSString *readableName = appBundle.infoDictionary[(__bridge NSString *)kCFBundleNameKey];
    return displayName ?: readableName;
}

String AutomationClient::browserVersion() const
{
    if (m_delegateMethods.browserVersionForAutomation)
        return [m_delegate _processPoolBrowserVersionForAutomation:m_processPool];

    // Fall back to using the app short version (i.e., '11.1.1').
    NSBundle *appBundle = [NSBundle mainBundle];
    return appBundle.infoDictionary[(__bridge NSString *)_kCFBundleShortVersionStringKey];
}

} // namespace WebKit

#endif // ENABLE(REMOTE_INSPECTOR)
