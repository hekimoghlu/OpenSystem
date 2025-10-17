/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
#include "WebExtensionControllerConfiguration.h"

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

WebExtensionControllerConfiguration::WebExtensionControllerConfiguration(IsPersistent persistent)
    : m_storageDirectory(persistent == IsPersistent::Yes ? createStorageDirectoryPath() : nullString())
{
}

WebExtensionControllerConfiguration::WebExtensionControllerConfiguration(TemporaryTag, const String& storageDirectory)
    : m_temporary(true)
    , m_storageDirectory(!storageDirectory.isEmpty() ? storageDirectory : createTemporaryStorageDirectoryPath())
{
}

WebExtensionControllerConfiguration::WebExtensionControllerConfiguration(const WTF::UUID& identifier)
    : m_identifier(identifier)
    , m_storageDirectory(createStorageDirectoryPath(identifier))
{
}

#if PLATFORM(COCOA)
bool WebExtensionControllerConfiguration::operator==(const WebExtensionControllerConfiguration& other) const
{
    return this == &other || (m_identifier == other.m_identifier && m_storageDirectory == other.m_storageDirectory && m_webViewConfiguration == other.m_webViewConfiguration && m_defaultWebsiteDataStore == other.m_defaultWebsiteDataStore);
}
#endif

WebsiteDataStore& WebExtensionControllerConfiguration::defaultWebsiteDataStore() const
{
    if (m_defaultWebsiteDataStore)
        return *m_defaultWebsiteDataStore;
    return WebsiteDataStore::defaultDataStore();
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
