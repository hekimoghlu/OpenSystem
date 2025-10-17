/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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

#if ENABLE(APPLICATION_MANIFEST)

#include "ApplicationManifest.h"
#include <optional>
#include <wtf/JSONValues.h>

namespace WebCore {

class Color;
class Document;

class ApplicationManifestParser {
public:
    WEBCORE_EXPORT static ApplicationManifest parse(Document&, const String&, const URL& manifestURL, const URL& documentURL);
    WEBCORE_EXPORT static ApplicationManifest parse(const String&, const URL& manifestURL, const URL& documentURL);
    WEBCORE_EXPORT static std::optional<ApplicationManifest> parseWithValidation(const String&, const URL& manifestURL, const URL& documentURL);

private:
    ApplicationManifestParser(RefPtr<Document>);
    ApplicationManifest parseManifest(const JSON::Object&, const String&, const URL&, const URL&);

    RefPtr<JSON::Object> createJSONObject(const String&);
    URL parseStartURL(const JSON::Object&, const URL&);
    ApplicationManifest::Direction parseDir(const JSON::Object&);
    ApplicationManifest::Display parseDisplay(const JSON::Object&);
    const std::optional<ScreenOrientationLockType> parseOrientation(const JSON::Object&);
    String parseName(const JSON::Object&);
    String parseDescription(const JSON::Object&);
    String parseShortName(const JSON::Object&);
    std::optional<URL> parseScope(const JSON::Object&, const URL&, const URL&);
    Vector<String> parseCategories(const JSON::Object&);
    Vector<ApplicationManifest::Icon> parseIcons(const JSON::Object&);
    Vector<ApplicationManifest::Shortcut> parseShortcuts(const JSON::Object&);
    URL parseId(const JSON::Object&, const URL&);

    Color parseColor(const JSON::Object&, const String& propertyName);
    String parseGenericString(const JSON::Object&, const String&);

    void logManifestPropertyNotAString(const String&);
    void logManifestPropertyInvalidURL(const String&);
    void logDeveloperWarning(const String& message);

    RefPtr<Document> m_document;
    URL m_manifestURL;
};

} // namespace WebCore

#endif // ENABLE(APPLICATION_MANIFEST)
