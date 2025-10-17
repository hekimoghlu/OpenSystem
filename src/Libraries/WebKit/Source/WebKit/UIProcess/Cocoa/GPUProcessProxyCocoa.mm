/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#include "config.h"
#include "GPUProcessProxy.h"

#if ENABLE(GPU_PROCESS)

#include "ArgumentCodersCocoa.h"
#include "GPUProcessCreationParameters.h"
#include "GPUProcessMessages.h"
#include "MediaPermissionUtilities.h"
#include "WebProcessProxy.h"
#include <wtf/cocoa/SpanCocoa.h>

#if HAVE(POWERLOG_TASK_MODE_QUERY)
#include <pal/spi/mac/PowerLogSPI.h>
#include <wtf/darwin/WeakLinking.h>

WTF_WEAK_LINK_FORCE_IMPORT(PLQueryRegistered);
#endif

namespace WebKit {

bool GPUProcessProxy::s_enableMetalDebugDeviceInNewGPUProcessesForTesting { false };
bool GPUProcessProxy::s_enableMetalShaderValidationInNewGPUProcessesForTesting { false };

void GPUProcessProxy::platformInitializeGPUProcessParameters(GPUProcessCreationParameters& parameters)
{
    parameters.mobileGestaltExtensionHandle = createMobileGestaltSandboxExtensionIfNeeded();
    parameters.gpuToolsExtensionHandles = createGPUToolsSandboxExtensionHandlesIfNeeded();
    parameters.applicationVisibleName = applicationVisibleName();
#if PLATFORM(MAC)
    if (auto launchServicesExtensionHandle = SandboxExtension::createHandleForMachLookup("com.apple.coreservices.launchservicesd"_s, std::nullopt))
        parameters.launchServicesExtensionHandle = WTFMove(*launchServicesExtensionHandle);
#endif
    parameters.enableMetalDebugDeviceForTesting = m_isMetalDebugDeviceEnabledForTesting;
    parameters.enableMetalShaderValidationForTesting = m_isMetalShaderValidationEnabledForTesting;
}

#if HAVE(POWERLOG_TASK_MODE_QUERY)
bool GPUProcessProxy::isPowerLoggingInTaskMode()
{
    CFDictionaryRef dictionary = nullptr;
    if (PLQueryRegistered)
        dictionary = PLQueryRegistered(PLClientIDWebKit, CFSTR("TaskModeQuery"), nullptr);
    if (!dictionary)
        return false;
    CFNumberRef taskModeRef = static_cast<CFNumberRef>(CFDictionaryGetValue(dictionary, CFSTR("Task Mode")));
    if (!taskModeRef)
        return false;
    int taskMode = 0;
    if (!CFNumberGetValue(taskModeRef, kCFNumberIntType, &taskMode))
        return false;
    return !!taskMode;
}

void GPUProcessProxy::enablePowerLogging()
{
    RELEASE_LOG(Sandbox, "GPUProcessProxy::enablePowerLogging()");
    auto handle = SandboxExtension::createHandleForMachLookup("com.apple.powerlog.plxpclogger.xpc"_s, std::nullopt);
    if (!handle)
        return;
    send(Messages::GPUProcess::EnablePowerLogging(WTFMove(*handle)), 0);
}
#endif // HAVE(POWERLOG_TASK_MODE_QUERY)

Vector<SandboxExtension::Handle> GPUProcessProxy::createGPUToolsSandboxExtensionHandlesIfNeeded()
{
    if (!WebProcessProxy::shouldEnableRemoteInspector())
        return { };

    return SandboxExtension::createHandlesForMachLookup({ "com.apple.gputools.service"_s, }, std::nullopt);
}

#if USE(EXTENSIONKIT)
void GPUProcessProxy::sendBookmarkDataForCacheDirectory()
{
    Ref protectedConnection = connection();
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), makeBlockPtr([protectedConnection = WTFMove(protectedConnection)] () mutable {
        NSError *error = nil;
        RetainPtr directoryURL = [[NSFileManager defaultManager] URLForDirectory:NSLibraryDirectory inDomain:NSUserDomainMask appropriateForURL:nil create:NO error:&error];
        RetainPtr url = adoptNS([[NSURL alloc] initFileURLWithPath:@"Caches/com.apple.WebKit.GPU/" relativeToURL:directoryURL.get()]);
        error = nil;
        RetainPtr bookmark = [url bookmarkDataWithOptions:NSURLBookmarkCreationMinimalBookmark includingResourceValuesForKeys:nil relativeToURL:nil error:&error];
        protectedConnection->send(Messages::GPUProcess::ResolveBookmarkDataForCacheDirectory(span(bookmark.get())), 0);
    }).get());
}
#endif

}

#endif
