/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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

#if ENABLE(GPU_PROCESS)

#include "AuxiliaryProcessCreationParameters.h"
#include "SandboxExtension.h"
#include <wtf/ProcessID.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

struct GPUProcessCreationParameters {
    AuxiliaryProcessCreationParameters auxiliaryProcessParameters;
#if ENABLE(MEDIA_STREAM)
    bool useMockCaptureDevices { false };
#if PLATFORM(MAC)
    SandboxExtension::Handle microphoneSandboxExtensionHandle;
    SandboxExtension::Handle launchServicesExtensionHandle;
#endif
#endif
#if USE(MODERN_AVCONTENTKEYSESSION)
    bool shouldUseModernAVContentKeySession { false };
#endif

#if USE(SANDBOX_EXTENSIONS_FOR_CACHE_AND_TEMP_DIRECTORY_ACCESS)
    SandboxExtension::Handle containerCachesDirectoryExtensionHandle;
    SandboxExtension::Handle containerTemporaryDirectoryExtensionHandle;
    String containerCachesDirectory;
#endif
#if PLATFORM(IOS_FAMILY)
    Vector<SandboxExtension::Handle> compilerServiceExtensionHandles;
    Vector<SandboxExtension::Handle> dynamicIOKitExtensionHandles;
#endif
    std::optional<SandboxExtension::Handle> mobileGestaltExtensionHandle;
#if PLATFORM(COCOA) && ENABLE(REMOTE_INSPECTOR)
    Vector<SandboxExtension::Handle> gpuToolsExtensionHandles;
#endif

    String applicationVisibleName;

#if USE(GBM)
    String renderDeviceFile;
#endif
    Vector<String> overrideLanguages;
#if PLATFORM(COCOA)
    bool enableMetalDebugDeviceForTesting { false };
    bool enableMetalShaderValidationForTesting { false };
#endif
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
