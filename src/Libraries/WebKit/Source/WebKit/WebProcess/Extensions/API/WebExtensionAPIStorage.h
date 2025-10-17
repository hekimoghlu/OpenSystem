/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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

#include "JSWebExtensionAPIStorage.h"
#include "JSWebExtensionWrappable.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionAPIStorageArea.h"

namespace WebKit {

class WebExtensionAPIStorageArea;

class WebExtensionAPIStorage : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIStorage, storage, storage);

public:
#if PLATFORM(COCOA)
    bool isPropertyAllowed(const ASCIILiteral& propertyName, WebPage*);

    WebExtensionAPIStorageArea& local();
    WebExtensionAPIStorageArea& session();
    WebExtensionAPIStorageArea& sync();

    WebExtensionAPIEvent& onChanged();

private:
    friend class WebExtensionContextProxy;

    WebExtensionAPIStorageArea& storageAreaForType(WebExtensionDataType);

    RefPtr<WebExtensionAPIStorageArea> m_local;
    RefPtr<WebExtensionAPIStorageArea> m_session;
    RefPtr<WebExtensionAPIStorageArea> m_sync;

    RefPtr<WebExtensionAPIEvent> m_onChanged;
#endif
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
