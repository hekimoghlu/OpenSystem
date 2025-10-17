/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionAPIStorage.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "WebExtensionAPINamespace.h"
#import "WebExtensionAPIStorageArea.h"
#import "WebExtensionContextMessages.h"
#import "WebExtensionContextProxy.h"

namespace WebKit {

bool WebExtensionAPIStorage::isPropertyAllowed(const ASCIILiteral& propertyName, WebPage*)
{
    if (UNLIKELY(extensionContext().isUnsupportedAPI(propertyPath(), propertyName)))
        return false;

    if (propertyName == "session"_s)
        return extensionContext().isSessionStorageAllowedInContentScripts() || isForMainWorld();

    ASSERT_NOT_REACHED();
    return false;
}

WebExtensionAPIStorageArea& WebExtensionAPIStorage::local()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/storage/local

    if (!m_local)
        m_local = WebExtensionAPIStorageArea::create(*this, WebExtensionDataType::Local);

    return *m_local;
}

WebExtensionAPIStorageArea& WebExtensionAPIStorage::session()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/storage/session

    if (!m_session)
        m_session = WebExtensionAPIStorageArea::create(*this, WebExtensionDataType::Session);

    return *m_session;
}

WebExtensionAPIStorageArea& WebExtensionAPIStorage::sync()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/storage/sync

    if (!m_sync)
        m_sync = WebExtensionAPIStorageArea::create(*this, WebExtensionDataType::Sync);

    return *m_sync;
}

WebExtensionAPIStorageArea& WebExtensionAPIStorage::storageAreaForType(WebExtensionDataType storageType)
{
    switch (storageType) {
    case WebExtensionDataType::Local:
        return local();
    case WebExtensionDataType::Session:
        return session();
    case WebExtensionDataType::Sync:
        return sync();
    }

    ASSERT_NOT_REACHED();
    return local();
}

WebExtensionAPIEvent& WebExtensionAPIStorage::onChanged()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/storage/onChanged

    if (!m_onChanged)
        m_onChanged = WebExtensionAPIEvent::create(*this, WebExtensionEventListenerType::StorageOnChanged);

    return *m_onChanged;
}

void WebExtensionContextProxy::dispatchStorageChangedEvent(const String& changesJSON, WebExtensionDataType dataType, WebExtensionContentWorldType contentWorldType)
{
    if (!hasDOMWrapperWorld(contentWorldType))
        return;

    id changes = parseJSON(changesJSON);
    auto areaName = toAPIString(dataType);

    enumerateFramesAndNamespaceObjects([&](WebFrame&, auto& namespaceObject) {
        namespaceObject.storage().onChanged().invokeListenersWithArgument(changes, areaName);
        namespaceObject.storage().storageAreaForType(dataType).onChanged().invokeListenersWithArgument(changes, areaName);
    }, toDOMWrapperWorld(contentWorldType));
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
