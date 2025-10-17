/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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

#if ENABLE(GPU_PROCESS)

#include "GPUProcessCreationParameters.h"

#if USE(WPE_RENDERER)
#include <WebCore/PlatformDisplayLibWPE.h>
#endif

namespace WebKit {

void GPUProcess::platformInitializeGPUProcess(GPUProcessCreationParameters&)
{
#if USE(WPE_RENDERER)
    WebCore::PlatformDisplay::setSharedDisplay(WebCore::PlatformDisplayLibWPE::create(0));
#endif
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

#endif // ENABLE(GPU_PROCESS)
