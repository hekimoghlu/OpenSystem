/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
#include "WebProcess.h"

#include "LogInitialization.h"
#include "WebProcessCreationParameters.h"
#include <WebCore/LogInitialization.h>
#include <wtf/LogInitialization.h>

#if USE(WPE_RENDERER)
#include <WebCore/PlatformDisplayLibWPE.h>
#endif

namespace WebKit {

void WebProcess::platformInitializeWebProcess(WebProcessCreationParameters& parameters)
{
#if USE(WPE_RENDERER)
    if (!parameters.isServiceWorkerProcess)
        WebCore::PlatformDisplay::setSharedDisplay(WebCore::PlatformDisplayLibWPE::create(parameters.hostClientFileDescriptor.release()));
#endif
    applyProcessCreationParameters(parameters.auxiliaryProcessParameters);
}

void WebProcess::platformInitializeProcess(const AuxiliaryProcessInitializationParameters&)
{
}

void WebProcess::platformSetWebsiteDataStoreParameters(WebProcessDataStoreParameters&&)
{
}

void WebProcess::platformTerminate()
{
}

void WebProcess::platformSetCacheModel(CacheModel)
{
}

void WebProcess::grantAccessToAssetServices(Vector<WebKit::SandboxExtension::Handle>&&)
{
}

void WebProcess::revokeAccessToAssetServices()
{
}

void WebProcess::switchFromStaticFontRegistryToUserFontRegistry(Vector<WebKit::SandboxExtension::Handle>&&)
{
}

} // namespace WebKit
