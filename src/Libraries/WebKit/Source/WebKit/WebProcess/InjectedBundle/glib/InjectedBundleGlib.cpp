/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
#include "InjectedBundle.h"

#include "WKBundleAPICast.h"
#include "WKBundleInitialize.h"
#include <wtf/FileSystem.h>
#include <wtf/text/CString.h>

namespace WebKit {

bool InjectedBundle::initialize(const WebProcessCreationParameters&, RefPtr<API::Object>&& initializationUserData)
{
    m_platformBundle = g_module_open(FileSystem::fileSystemRepresentation(m_path).data(), G_MODULE_BIND_LOCAL);
    if (!m_platformBundle) {
        g_warning("Error loading the injected bundle (%s): %s", m_path.utf8().data(), g_module_error());
        return false;
    }

    WKBundleInitializeFunctionPtr initializeFunction = 0;
    if (!g_module_symbol(m_platformBundle, "WKBundleInitialize", reinterpret_cast<void**>(&initializeFunction)) || !initializeFunction) {
        g_warning("Error loading WKBundleInitialize symbol from injected bundle.");
        return false;
    }

    initializeFunction(toAPI(this), toAPI(initializationUserData.get()));
    return true;
}

void InjectedBundle::setBundleParameter(WTF::String const&, std::span<const uint8_t>)
{
}

void InjectedBundle::setBundleParameters(std::span<const uint8_t>)
{
}

} // namespace WebKit
