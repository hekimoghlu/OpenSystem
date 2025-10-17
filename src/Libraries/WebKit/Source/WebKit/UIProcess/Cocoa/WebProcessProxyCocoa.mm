/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#import "WebProcessProxy.h"

#import "AccessibilitySupportSPI.h"
#import "CodeSigning.h"
#import "CoreIPCAuditToken.h"
#import "DefaultWebBrowserChecks.h"
#import "Logging.h"
#import "SandboxUtilities.h"
#import "SharedBufferReference.h"
#import "WKAPICast.h"
#import "WKBrowsingContextControllerInternal.h"
#import "WKBrowsingContextHandleInternal.h"
#import "WebProcessMessages.h"
#import "WebProcessPool.h"
#import <WebCore/ActivityState.h>
#import <pal/spi/ios/MobileGestaltSPI.h>
#import <sys/sysctl.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/RuntimeApplicationChecks.h>
#import <wtf/Scope.h>
#import <wtf/cocoa/Entitlements.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/cocoa/VectorCocoa.h>
#import <wtf/spi/darwin/SandboxSPI.h>

#if PLATFORM(IOS_FAMILY)
#import "AccessibilitySupportSPI.h"
#endif

#if ENABLE(REMOTE_INSPECTOR)
#import "WebInspectorUtilities.h"
#import <JavaScriptCore/RemoteInspectorConstants.h>
#endif

#if HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK)
#import <WebCore/CaptionUserPreferencesMediaAF.h>
#endif

#if PLATFORM(MAC) || PLATFORM(MACCATALYST)
#import "WindowServerConnection.h"
#endif

#if PLATFORM(MAC) || PLATFORM(MACCATALYST)
#import "TCCSoftLink.h"
#endif

namespace WebKit {

static const Seconds unexpectedActivityDuration = 10_s;

const MemoryCompactLookupOnlyRobinHoodHashSet<String>& WebProcessProxy::platformPathsWithAssumedReadAccess()
{
    static NeverDestroyed<MemoryCompactLookupOnlyRobinHoodHashSet<String>> platformPathsWithAssumedReadAccess(std::initializer_list<String> {
        [NSBundle bundleWithIdentifier:@"com.apple.WebCore"].resourcePath.stringByStandardizingPath,
        [NSBundle bundleForClass:NSClassFromString(@"WKWebView")].resourcePath.stringByStandardizingPath
    });

    return platformPathsWithAssumedReadAccess;
}

static Vector<String>& mediaTypeCache()
{
    ASSERT(RunLoop::isMain());
    static NeverDestroyed<Vector<String>> typeCache;
    return typeCache;
}

void WebProcessProxy::cacheMediaMIMETypes(const Vector<String>& types)
{
    if (!mediaTypeCache().isEmpty())
        return;

    mediaTypeCache() = types;
    for (Ref process : processPool().processes()) {
        if (process.ptr() != this)
            cacheMediaMIMETypesInternal(types);
    }
}

void WebProcessProxy::cacheMediaMIMETypesInternal(const Vector<String>& types)
{
    if (!mediaTypeCache().isEmpty())
        return;

    mediaTypeCache() = types;
    send(Messages::WebProcess::SetMediaMIMETypes(types), 0);
}

Vector<String> WebProcessProxy::mediaMIMETypes() const
{
    return mediaTypeCache();
}

#if ENABLE(REMOTE_INSPECTOR)
bool WebProcessProxy::shouldEnableRemoteInspector()
{
#if PLATFORM(IOS_FAMILY)
    return CFPreferencesGetAppIntegerValue(WIRRemoteInspectorEnabledKey, WIRRemoteInspectorDomainName, nullptr);
#else
    return CFPreferencesGetAppIntegerValue(CFSTR("ShowDevelopMenu"), bundleIdentifierForSandboxBroker(), nullptr);
#endif
}

void WebProcessProxy::enableRemoteInspectorIfNeeded()
{
    if (!shouldEnableRemoteInspector())
        return;
    send(Messages::WebProcess::EnableRemoteWebInspector(), 0);
}
#endif

#if HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK)
void WebProcessProxy::setCaptionDisplayMode(WebCore::CaptionUserPreferences::CaptionDisplayMode displayMode)
{
    WebCore::CaptionUserPreferencesMediaAF::platformSetCaptionDisplayMode(displayMode);
}

void WebProcessProxy::setCaptionLanguage(const String& language)
{
    WebCore::CaptionUserPreferencesMediaAF::platformSetPreferredLanguage(language);
}
#endif

void WebProcessProxy::unblockAccessibilityServerIfNeeded()
{
    if (m_hasSentMessageToUnblockAccessibilityServer)
        return;
#if PLATFORM(IOS_FAMILY)
    if (!_AXSApplicationAccessibilityEnabled())
        return;
#endif
    if (!processID())
        return;
    if (!canSendMessage())
        return;

    Vector<SandboxExtension::Handle> handleArray;
#if PLATFORM(IOS_FAMILY)
    handleArray = SandboxExtension::createHandlesForMachLookup({ }, auditToken(), SandboxExtension::MachBootstrapOptions::EnableMachBootstrap);
#endif

    send(Messages::WebProcess::UnblockServicesRequiredByAccessibility(WTFMove(handleArray)), 0);
    m_hasSentMessageToUnblockAccessibilityServer = true;
}

#if PLATFORM(MAC) || PLATFORM(MACCATALYST)
void WebProcessProxy::isAXAuthenticated(CoreIPCAuditToken&& auditToken, CompletionHandler<void(bool)>&& completionHandler)
{
    auto authenticated = TCCAccessCheckAuditToken(get_TCC_kTCCServiceAccessibility(), auditToken.auditToken(), nullptr);
    completionHandler(authenticated);
}
#endif

#if PLATFORM(MAC) || PLATFORM(MACCATALYST)
void WebProcessProxy::hardwareConsoleStateChanged()
{
    m_isConnectedToHardwareConsole = WindowServerConnection::singleton().hardwareConsoleState() == WindowServerConnection::HardwareConsoleState::Connected;
    for (const auto& page : m_pageMap.values())
        page->activityStateDidChange(WebCore::ActivityState::IsConnectedToHardwareConsole);
}
#endif

#if HAVE(AUDIO_COMPONENT_SERVER_REGISTRATIONS)
void WebProcessProxy::sendAudioComponentRegistrations()
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), [weakThis = WeakPtr { *this }] () mutable {

        auto registrations = fetchAudioComponentServerRegistrations();
        if (!registrations)
            return;
        
        RunLoop::protectedMain()->dispatch([weakThis = WTFMove(weakThis), registrations = WTFMove(registrations)] () mutable {
            if (!weakThis)
                return;

            weakThis->send(Messages::WebProcess::ConsumeAudioComponentRegistrations(IPC::SharedBufferReference(WTFMove(registrations))), 0);
        });
    });
}
#endif

bool WebProcessProxy::messageSourceIsValidWebContentProcess()
{
    if (!hasConnection()) {
        ASSERT_NOT_REACHED();
        return false;
    }

#if USE(APPLE_INTERNAL_SDK)
#if PLATFORM(IOS) || PLATFORM(VISION)
    // FIXME(rdar://80908833): On iOS, we can only perform the below checks for platform binaries until rdar://80908833 is fixed.
    if (!currentProcessIsPlatformBinary())
        return true;
#endif

    // WebKitTestRunner does not pass the isPlatformBinary check, we should return early in this case.
    if (isRunningTest(applicationBundleIdentifier()))
        return true;

    // Confirm that the connection is from a WebContent process:
    auto [signingIdentifier, isPlatformBinary] = codeSigningIdentifierAndPlatformBinaryStatus(connection().xpcConnection());

    if (!isPlatformBinary || !signingIdentifier.startsWith("com.apple.WebKit.WebContent"_s)) {
        RELEASE_LOG_ERROR(Process, "Process is not an entitled WebContent process.");
        return false;
    }
#endif

    return true;
}

std::optional<audit_token_t> WebProcessProxy::auditToken() const
{
    if (!hasConnection())
        return std::nullopt;
    
    return protectedConnection()->getAuditToken();
}

std::optional<Vector<SandboxExtension::Handle>> WebProcessProxy::fontdMachExtensionHandles()
{
    if (std::exchange(m_sentFontdMachExtensionHandles, true))
        return std::nullopt;
    return SandboxExtension::createHandlesForMachLookup({ "com.apple.fonts"_s }, auditToken(), SandboxExtension::MachBootstrapOptions::EnableMachBootstrap);
}

#if USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/WebProcessProxyCocoaAdditions.mm>)
#import <WebKitAdditions/WebProcessProxyCocoaAdditions.mm>
#else
bool WebProcessProxy::shouldDisableJITCage() const
{
    return false;
}
#endif

#if ENABLE(LOGD_BLOCKING_IN_WEBCONTENT)
void WebProcessProxy::setupLogStream(uint32_t pid, IPC::StreamServerConnectionHandle&& serverConnection, LogStreamIdentifier logStreamIdentifier, CompletionHandler<void(IPC::Semaphore& streamWakeUpSemaphore, IPC::Semaphore& streamClientWaitSemaphore)>&& completionHandler)
{
    Ref logStream = LogStream::create(processID());
    logStream->setup(WTFMove(serverConnection), logStreamIdentifier, WTFMove(completionHandler));
    m_logStream = WTFMove(logStream);
}
#endif // ENABLE(LOGD_BLOCKING_IN_WEBCONTENT)

}

