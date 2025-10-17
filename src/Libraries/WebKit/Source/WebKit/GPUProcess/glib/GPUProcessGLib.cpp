/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
#include "GPUProcess.h"

#if ENABLE(GPU_PROCESS) && (PLATFORM(GTK) || PLATFORM(WPE))

#include "GPUProcessCreationParameters.h"

#if USE(GBM)
#include <WebCore/DRMDeviceManager.h>
#include <WebCore/PlatformDisplayGBM.h>
#endif

namespace WebKit {

void GPUProcess::platformInitializeGPUProcess(GPUProcessCreationParameters& parameters)
{
#if USE(GBM)
    WebCore::DRMDeviceManager::singleton().initializeMainDevice(parameters.renderDeviceFile);

    if (auto* device = WebCore::DRMDeviceManager::singleton().mainGBMDeviceNode(WebCore::DRMDeviceManager::NodeType::Render)) {
        WebCore::PlatformDisplay::setSharedDisplay(WebCore::PlatformDisplayGBM::create(device));
        return;
    }
#else
    UNUSED_PARAM(parameters);
#endif

    WTFLogAlways("Could not create EGL display for GPU process: no supported platform available. Aborting...");
    CRASH();
}

void GPUProcess::initializeProcess(const AuxiliaryProcessInitializationParameters&)
{
}

void GPUProcess::initializeProcessName(const AuxiliaryProcessInitializationParameters&)
{
}

void GPUProcess::initializeSandbox(const AuxiliaryProcessInitializationParameters&, SandboxInitializationParameters&)
{
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && (PLATFORM(GTK) || PLATFORM(WPE))
