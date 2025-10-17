/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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

#include <WebCore/PlatformLocale.h>
#include <wtf/Forward.h>
#include <wtf/JSONValues.h>
#include <wtf/Noncopyable.h>

namespace JSC { namespace Yarr {
class RegularExpression;
} }

namespace WebKit {
class WebExtension;

class WebExtensionLocalization : public RefCounted<WebExtensionLocalization> {
    WTF_MAKE_NONCOPYABLE(WebExtensionLocalization)

public:
    template<typename... Args>
    static Ref<WebExtensionLocalization> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionLocalization(std::forward<Args>(args)...));
    }

    explicit WebExtensionLocalization(WebKit::WebExtension&);
    explicit WebExtensionLocalization(RefPtr<JSON::Object> localizedJSON, const String& uniqueIdentifier);
    explicit WebExtensionLocalization(RefPtr<JSON::Object> regionalLocalization, RefPtr<JSON::Object> languageLocalization, RefPtr<JSON::Object> defaultLocalization, const String& withBestLocale, const String& uniqueIdentifier);

    const String& uniqueIdentifier() { return m_uniqueIdentifier; };
    RefPtr<JSON::Object> localizationJSON() { return m_localizationJSON; };

    RefPtr<JSON::Object> localizedJSONforJSON(RefPtr<JSON::Object>);
    String localizedStringForKey(String key, Vector<String> placeholders = { });
    String localizedStringForString(String);

private:
    void loadRegionalLocalization(RefPtr<JSON::Object> regionalLocalization, RefPtr<JSON::Object> languageLocalization, RefPtr<JSON::Object> defaultLocalization, const String& withBestLocale = { }, const String& uniqueIdentifier = { });

    RefPtr<JSON::Object> localizationJSONForWebExtension(WebKit::WebExtension&, const String& withLocale);
    RefPtr<JSON::Array> localizedArrayForArray(RefPtr<JSON::Array>);
    Ref<JSON::Object> predefinedMessages();

    String stringByReplacingNamedPlaceholdersInString(String sourceString, RefPtr<JSON::Object> placeholders);
    String stringByReplacingPositionalPlaceholdersInString(String sourceString, Vector<String> placeholders = { });

    std::unique_ptr<WebCore::Locale> m_locale;
    String m_localeString;
    String m_uniqueIdentifier;
    RefPtr<JSON::Object> m_localizationJSON;
};

}
