/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
#pragma once

#import "config.h"
#import "GPUProcess.h"

#if ENABLE(GPU_PROCESS) && PLATFORM(COCOA)

#import "GPUConnectionToWebProcess.h"
#import "GPUProcessCreationParameters.h"
#import "Logging.h"
#import "RemoteRenderingBackend.h"
#import <pal/spi/cocoa/AVFoundationSPI.h>
#import <pal/spi/cocoa/MetalSPI.h>
#import <wtf/RetainPtr.h>
#import <wtf/cocoa/SpanCocoa.h>

#if PLATFORM(MAC)
#include <pal/spi/cocoa/LaunchServicesSPI.h>
#endif

#if PLATFORM(VISION) && ENABLE(MODEL_PROCESS)
#include "CoreIPCAuditToken.h"
#include "SharedFileHandle.h"
#include "WKSharedSimulationConnectionHelper.h"
#endif

#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebKit {
using namespace WebCore;

#if USE(OS_STATE)

RetainPtr<NSDictionary> GPUProcess::additionalStateForDiagnosticReport() const
{
    auto stateDictionary = adoptNS([[NSMutableDictionary alloc] initWithCapacity:1]);
    if (!m_webProcessConnections.isEmpty()) {
        auto webProcessConnectionInfo = adoptNS([[NSMutableDictionary alloc] initWithCapacity:m_webProcessConnections.size()]);
        for (auto& identifierAndConnection : m_webProcessConnections) {
            auto& [webProcessIdentifier, connection] = identifierAndConnection;
            auto& backendMap = connection->remoteRenderingBackendMap();
            if (backendMap.isEmpty())
                continue;

            auto stateInfo = adoptNS([[NSMutableDictionary alloc] initWithCapacity:backendMap.size()]);
            // FIXME: Log some additional diagnostic state on RemoteRenderingBackend.
            [webProcessConnectionInfo setObject:stateInfo.get() forKey:webProcessIdentifier.loggingString()];
        }

        if ([webProcessConnectionInfo count])
            [stateDictionary setObject:webProcessConnectionInfo.get() forKey:@"RemoteRenderingBackend states"];
    }
    return stateDictionary;
}

#endif // USE(OS_STATE)

#if ENABLE(CFPREFS_DIRECT_MODE)
void GPUProcess::dispatchSimulatedNotificationsForPreferenceChange(const String& key)
{
}
#endif // ENABLE(CFPREFS_DIRECT_MODE)

#if ENABLE(MEDIA_STREAM)
void GPUProcess::ensureAVCaptureServerConnection()
{
    RELEASE_LOG(WebRTC, "GPUProcess::ensureAVCaptureServerConnection: Entering.");
#if HAVE(AVCAPTUREDEVICE) && HAVE(AVSAMPLEBUFFERVIDEOOUTPUT)
    if ([PAL::getAVCaptureDeviceClass() respondsToSelector:@selector(ensureServerConnection)]) {
        RELEASE_LOG(WebRTC, "GPUProcess::ensureAVCaptureServerConnection: Calling [AVCaptureDevice ensureServerConnection]");
        [PAL::getAVCaptureDeviceClass() ensureServerConnection];
    }
#endif
}
#endif

void GPUProcess::platformInitializeGPUProcess(GPUProcessCreationParameters& parameters)
{
#if PLATFORM(MAC)
    auto launchServicesExtension = SandboxExtension::create(WTFMove(parameters.launchServicesExtensionHandle));
    if (launchServicesExtension) {
        bool ok = launchServicesExtension->consume();
        ASSERT_UNUSED(ok, ok);
    }

    // It is important to check in with launch services before setting the process name.
    launchServicesCheckIn();

    // Update process name while holding the Launch Services sandbox extension
    updateProcessName();

    // Close connection to launch services.
#if HAVE(HAVE_LS_SERVER_CONNECTION_STATUS_RELEASE_NOTIFICATIONS_MASK)
    _LSSetApplicationLaunchServicesServerConnectionStatus(kLSServerConnectionStatusDoNotConnectToServerMask | kLSServerConnectionStatusReleaseNotificationsMask, nullptr);
#else
    _LSSetApplicationLaunchServicesServerConnectionStatus(kLSServerConnectionStatusDoNotConnectToServerMask, nullptr);
#endif

    if (launchServicesExtension)
        launchServicesExtension->revoke();
#endif // PLATFORM(MAC)

    if (parameters.enableMetalDebugDeviceForTesting) {
        RELEASE_LOG(Process, "%p - GPUProcess::platformInitializeGPUProcess: enabling Metal debug device", this);
        setenv("MTL_DEBUG_LAYER", "1", 1);
    }

    if (parameters.enableMetalShaderValidationForTesting) {
        RELEASE_LOG(Process, "%p - GPUProcess::platformInitializeGPUProcess: enabling Metal shader validation", this);
        setenv("MTL_SHADER_VALIDATION", "1", 1);
        setenv("MTL_SHADER_VALIDATION_ABORT_ON_FAULT", "1", 1);
        setenv("MTL_SHADER_VALIDATION_REPORT_TO_STDERR", "1", 1);
    }

#if USE(SANDBOX_EXTENSIONS_FOR_CACHE_AND_TEMP_DIRECTORY_ACCESS) && USE(EXTENSIONKIT)
    MTLSetShaderCachePath(parameters.containerCachesDirectory);
#endif
}

#if USE(EXTENSIONKIT)
void GPUProcess::resolveBookmarkDataForCacheDirectory(std::span<const uint8_t> bookmarkData)
{
    RetainPtr bookmark = toNSData(bookmarkData);
    BOOL bookmarkIsStale = NO;
    NSError* error = nil;
    [NSURL URLByResolvingBookmarkData:bookmark.get() options:NSURLBookmarkResolutionWithoutUI relativeToURL:nil bookmarkDataIsStale:&bookmarkIsStale error:&error];
}
#endif

#if PLATFORM(VISION) && ENABLE(MODEL_PROCESS)
void GPUProcess::requestSharedSimulationConnection(CoreIPCAuditToken&& modelProcessAuditToken, CompletionHandler<void(std::optional<IPC::SharedFileHandle>)>&& completionHandler)
{
    Ref<WKSharedSimulationConnectionHelper> sharedSimulationConnectionHelper = adoptRef(*new WKSharedSimulationConnectionHelper);
    sharedSimulationConnectionHelper->requestSharedSimulationConnectionForAuditToken(modelProcessAuditToken.auditToken(), [sharedSimulationConnectionHelper, completionHandler = WTFMove(completionHandler)] (RetainPtr<NSFileHandle> sharedSimulationConnection, RetainPtr<id> appService) mutable {
        if (!sharedSimulationConnection) {
            RELEASE_LOG_ERROR(ModelElement, "GPUProcess: Shared simulation join request failed");
            completionHandler(std::nullopt);
            return;
        }

        RELEASE_LOG(ModelElement, "GPUProcess: Shared simulation join request succeeded");
        completionHandler(IPC::SharedFileHandle::create([sharedSimulationConnection fileDescriptor]));
    });
}
#endif

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && PLATFORM(COCOA)
