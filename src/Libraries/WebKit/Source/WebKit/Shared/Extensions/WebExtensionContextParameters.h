/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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

#include "APIData.h"
#include "WebExtensionContext.h"
#include "WebExtensionContextIdentifier.h"
#include "WebExtensionTabIdentifier.h"
#include "WebExtensionWindowIdentifier.h"
#include <wtf/URL.h>

namespace WebKit {

struct WebExtensionContextParameters {
    WebExtensionContextIdentifier identifier;

    URL baseURL;
    String uniqueIdentifier;
    HashSet<String> unsupportedAPIs;

    HashMap<String, WallTime> grantedPermissions;

    RefPtr<API::Data> localizationJSON;
    RefPtr<API::Data> manifestJSON;

    double manifestVersion { 0 };
    bool isSessionStorageAllowedInContentScripts { false };

    std::optional<WebCore::PageIdentifier> backgroundPageIdentifier;
#if ENABLE(INSPECTOR_EXTENSIONS)
    Vector<WebExtensionContext::PageIdentifierTuple> inspectorPageIdentifiers;
    Vector<WebExtensionContext::PageIdentifierTuple> inspectorBackgroundPageIdentifiers;
#endif
    Vector<WebExtensionContext::PageIdentifierTuple> popupPageIdentifiers;
    Vector<WebExtensionContext::PageIdentifierTuple> tabPageIdentifiers;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
