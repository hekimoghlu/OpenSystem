/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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

#include "UserContentProvider.h"
#include "UserScriptTypes.h"
#include "UserStyleSheetTypes.h"

namespace WebCore {

class UserContentController final : public UserContentProvider {
public:
    WEBCORE_EXPORT static Ref<UserContentController> create();
    WEBCORE_EXPORT ~UserContentController();

    WEBCORE_EXPORT void addUserScript(DOMWrapperWorld&, std::unique_ptr<UserScript>);
    WEBCORE_EXPORT void removeUserScript(DOMWrapperWorld&, const URL&);
    WEBCORE_EXPORT void removeUserScripts(DOMWrapperWorld&);

    WEBCORE_EXPORT void addUserStyleSheet(DOMWrapperWorld&, std::unique_ptr<UserStyleSheet>, UserStyleInjectionTime);
    WEBCORE_EXPORT void removeUserStyleSheet(DOMWrapperWorld&, const URL&);
    WEBCORE_EXPORT void removeUserStyleSheets(DOMWrapperWorld&);

    WEBCORE_EXPORT void removeAllUserContent();

private:
    UserContentController();

    // UserContentProvider
    void forEachUserScript(Function<void(DOMWrapperWorld&, const UserScript&)>&&) const final;
    void forEachUserStyleSheet(Function<void(const UserStyleSheet&)>&&) const final;
#if ENABLE(USER_MESSAGE_HANDLERS)
    void forEachUserMessageHandler(Function<void(const UserMessageHandlerDescriptor&)>&&) const final;
#endif
#if ENABLE(CONTENT_EXTENSIONS)
    ContentExtensions::ContentExtensionsBackend& userContentExtensionBackend() override { return m_contentExtensionBackend; }
#endif

    UserScriptMap m_userScripts;
    UserStyleSheetMap m_userStyleSheets;
#if ENABLE(CONTENT_EXTENSIONS)
    ContentExtensions::ContentExtensionsBackend m_contentExtensionBackend;
#endif
};

} // namespace WebCore
