/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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
#import "GPUConnectionToWebProcess.h"

#if ENABLE(GPU_PROCESS)

#import "Logging.h"
#import "MediaPermissionUtilities.h"
#import <WebCore/LocalizedStrings.h>
#import <WebCore/RealtimeMediaSourceCenter.h>
#import <WebCore/RegistrableDomain.h>
#import <WebCore/SecurityOrigin.h>
#import <pal/spi/cocoa/LaunchServicesSPI.h>
#import <wtf/OSObjectPtr.h>

#if HAVE(SYSTEM_STATUS)
#import "SystemStatusSPI.h"
#import <pal/ios/SystemStatusSoftLink.h>
#endif

#import "TCCSoftLink.h"

namespace WebKit {

#if ENABLE(MEDIA_STREAM)
bool GPUConnectionToWebProcess::setCaptureAttributionString()
{
#if HAVE(SYSTEM_STATUS)
    if (![PAL::getSTDynamicActivityAttributionPublisherClass() respondsToSelector:@selector(setCurrentAttributionStringWithFormat:auditToken:)]
        && ![PAL::getSTDynamicActivityAttributionPublisherClass() respondsToSelector:@selector(setCurrentAttributionWebsiteString:auditToken:)]) {
        return true;
    }

    auto auditToken = gpuProcess().parentProcessConnection()->getAuditToken();
    if (!auditToken)
        return false;

    auto *visibleName = applicationVisibleNameFromOrigin(m_captureOrigin->data());
    if (!visibleName)
        visibleName = gpuProcess().applicationVisibleName();

    if ([PAL::getSTDynamicActivityAttributionPublisherClass() respondsToSelector:@selector(setCurrentAttributionWebsiteString:auditToken:)])
        [PAL::getSTDynamicActivityAttributionPublisherClass() setCurrentAttributionWebsiteString:visibleName auditToken:auditToken.value()];
    else {
        RetainPtr<NSString> formatString = [NSString stringWithFormat:WEB_UI_NSSTRING(@"%@ in %%@", "The domain and application using the camera and/or microphone. The first argument is domain, the second is the application name (iOS only)."), visibleName];
        [PAL::getSTDynamicActivityAttributionPublisherClass() setCurrentAttributionStringWithFormat:formatString.get() auditToken:auditToken.value()];
    }
#endif

    return true;
}
#endif // ENABLE(MEDIA_STREAM)

#if ENABLE(APP_PRIVACY_REPORT)
void GPUConnectionToWebProcess::setTCCIdentity()
{
#if !PLATFORM(MACCATALYST)
    auto auditToken = gpuProcess().parentProcessConnection()->getAuditToken();
    if (!auditToken) {
        RELEASE_LOG_ERROR(WebRTC, "getAuditToken returned null");
        return;
    }

    NSError *error = nil;
    auto bundleProxy = [LSBundleProxy bundleProxyWithAuditToken:*auditToken error:&error];
    RELEASE_LOG_ERROR_IF(error, WebRTC, "-[LSBundleProxy bundleProxyWithAuditToken:error:] failed with error %s", [[error localizedDescription] UTF8String]);

    String bundleIdentifier = bundleProxy.bundleIdentifier;
    if (bundleIdentifier.isNull())
        bundleIdentifier = m_applicationBundleIdentifier;

    if (bundleIdentifier.isNull()) {
        RELEASE_LOG_ERROR(WebRTC, "Unable to get the bundle identifier");
        return;
    }

    auto identity = adoptOSObject(tcc_identity_create(TCC_IDENTITY_CODE_BUNDLE_ID, bundleIdentifier.utf8().data()));
    if (!identity) {
        RELEASE_LOG_ERROR(WebRTC, "tcc_identity_create returned null");
        return;
    }

    WebCore::RealtimeMediaSourceCenter::singleton().setIdentity(WTFMove(identity));
#endif // !PLATFORM(MACCATALYST)
}
#endif // ENABLE(APP_PRIVACY_REPORT)

#if ENABLE(EXTENSION_CAPABILITIES)
String GPUConnectionToWebProcess::mediaEnvironment(WebCore::PageIdentifier pageIdentifier)
{
    return m_mediaEnvironments.get(pageIdentifier);
}

void GPUConnectionToWebProcess::setMediaEnvironment(WebCore::PageIdentifier pageIdentifier, const String& mediaEnvironment)
{
    if (mediaEnvironment.isEmpty())
        m_mediaEnvironments.remove(pageIdentifier);
    else
        m_mediaEnvironments.set(pageIdentifier, mediaEnvironment);
}
#endif

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
