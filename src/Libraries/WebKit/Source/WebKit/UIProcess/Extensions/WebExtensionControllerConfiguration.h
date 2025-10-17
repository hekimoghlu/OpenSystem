/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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

#if ENABLE(WK_WEB_EXTENSIONS)

#include "APIObject.h"
#include "WebsiteDataStore.h"
#include <wtf/Forward.h>
#include <wtf/Markable.h>
#include <wtf/RetainPtr.h>
#include <wtf/UUID.h>

OBJC_CLASS WKWebExtensionControllerConfiguration;
OBJC_CLASS WKWebViewConfiguration;

namespace WebKit {

class WebExtensionControllerConfiguration : public API::ObjectImpl<API::Object::Type::WebExtensionControllerConfiguration> {
    WTF_MAKE_NONCOPYABLE(WebExtensionControllerConfiguration);

public:
    enum class IsPersistent : bool { No, Yes };
    enum TemporaryTag { Temporary };

    static Ref<WebExtensionControllerConfiguration> createDefault() { return adoptRef(*new WebExtensionControllerConfiguration(IsPersistent::Yes)); }
    static Ref<WebExtensionControllerConfiguration> createNonPersistent() { return adoptRef(*new WebExtensionControllerConfiguration(IsPersistent::No)); }
    static Ref<WebExtensionControllerConfiguration> createTemporary() { return adoptRef(*new WebExtensionControllerConfiguration(Temporary)); }
    static Ref<WebExtensionControllerConfiguration> create(const WTF::UUID& identifier) { return adoptRef(*new WebExtensionControllerConfiguration(identifier)); }

    Ref<WebExtensionControllerConfiguration> copy() const;

    explicit WebExtensionControllerConfiguration(IsPersistent);
    explicit WebExtensionControllerConfiguration(TemporaryTag, const String& storageDirectory = nullString());
    explicit WebExtensionControllerConfiguration(const WTF::UUID&);

    std::optional<WTF::UUID> identifier() const { return m_identifier; }

    bool storageIsPersistent() const { return !m_storageDirectory.isEmpty(); }
    bool storageIsTemporary() const { return m_temporary; }

    const String& storageDirectory() const { return m_storageDirectory; }
    void setStorageDirectory(const String& directory) { m_storageDirectory = directory; }

#if PLATFORM(COCOA)
    WKWebViewConfiguration *webViewConfiguration();
    void setWebViewConfiguration(WKWebViewConfiguration *configuration) { m_webViewConfiguration = configuration; }
#endif

    WebsiteDataStore& defaultWebsiteDataStore() const;
    Ref<WebsiteDataStore> protectedDefaultWebsiteDataStore() const { return defaultWebsiteDataStore(); }
    void setDefaultWebsiteDataStore(WebsiteDataStore* dataStore) { m_defaultWebsiteDataStore = dataStore; }

    bool operator==(const WebExtensionControllerConfiguration&) const;

#ifdef __OBJC__
    WKWebExtensionControllerConfiguration *wrapper() const { return (WKWebExtensionControllerConfiguration *)API::ObjectImpl<API::Object::Type::WebExtensionControllerConfiguration>::wrapper(); }
#endif

private:
    static String createStorageDirectoryPath(std::optional<WTF::UUID> = std::nullopt);
    static String createTemporaryStorageDirectoryPath();

    Markable<WTF::UUID> m_identifier;
    bool m_temporary { false };
    String m_storageDirectory;
#if PLATFORM(COCOA)
    RetainPtr<WKWebViewConfiguration> m_webViewConfiguration;
#endif
    RefPtr<WebsiteDataStore> m_defaultWebsiteDataStore;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
