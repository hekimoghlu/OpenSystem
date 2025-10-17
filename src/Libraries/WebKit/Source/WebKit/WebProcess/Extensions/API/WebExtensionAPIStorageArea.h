/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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

#include "JSWebExtensionAPIStorageArea.h"
#include "JSWebExtensionWrappable.h"
#include "WebExtensionAPIEvent.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionDataType.h"
#include "WebPageProxyIdentifier.h"

namespace WebKit {

class WebExtensionAPIStorageArea : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIStorageArea, storageArea, storageArea);

public:
#if PLATFORM(COCOA)
    bool isPropertyAllowed(const ASCIILiteral& propertyName, WebPage*);

    void get(WebPageProxyIdentifier, id items, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getKeys(WebPageProxyIdentifier, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getBytesInUse(WebPageProxyIdentifier, id keys, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void set(WebPageProxyIdentifier, NSDictionary *items, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void remove(WebPageProxyIdentifier, id keys, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void clear(WebPageProxyIdentifier, Ref<WebExtensionCallbackHandler>&&);

    double quotaBytes();

    // Exposed only by storage.session.
    void setAccessLevel(WebPageProxyIdentifier, NSDictionary *, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

    // Exposed only by storage.sync.
    double quotaBytesPerItem();
    double maxItems();
    double maxWriteOperationsPerHour();
    double maxWriteOperationsPerMinute();

    WebExtensionAPIEvent& onChanged();

private:
    explicit WebExtensionAPIStorageArea(const WebExtensionAPIObject& parentObject, WebExtensionDataType type)
        : WebExtensionAPIObject(parentObject)
        , m_type(type)
    {
        setPropertyPath(toAPIString(type), &parentObject);
    }

    WebExtensionDataType m_type { WebExtensionDataType::Local };
    RefPtr<WebExtensionAPIEvent> m_onChanged;
#endif
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
