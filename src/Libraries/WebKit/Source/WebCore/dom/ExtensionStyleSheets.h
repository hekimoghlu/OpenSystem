/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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

#include "UserStyleSheet.h"
#include <memory>
#include <wtf/CheckedRef.h>
#include <wtf/RefPtr.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>
#include <wtf/text/WTFString.h>

#if ENABLE(CONTENT_EXTENSIONS)
#include "ContentExtensionStyleSheet.h"
#endif

namespace WebCore {

class CSSStyleSheet;
class Document;
class Node;
class StyleSheet;
class StyleSheetContents;
class StyleSheetList;
class WeakPtrImplWithEventTargetData;

class ExtensionStyleSheets final : public CanMakeCheckedPtr<ExtensionStyleSheets> {
    WTF_MAKE_TZONE_ALLOCATED(ExtensionStyleSheets);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ExtensionStyleSheets);
public:
    explicit ExtensionStyleSheets(Document&);
    ~ExtensionStyleSheets();

    CSSStyleSheet* pageUserSheet();
    const Vector<RefPtr<CSSStyleSheet>>& documentUserStyleSheets() const { return m_userStyleSheets; }
    const Vector<RefPtr<CSSStyleSheet>>& injectedUserStyleSheets() const;
    const Vector<RefPtr<CSSStyleSheet>>& injectedAuthorStyleSheets() const;
    const Vector<RefPtr<CSSStyleSheet>>& authorStyleSheetsForTesting() const { return m_authorStyleSheetsForTesting; }

    void clearPageUserSheet();
    void updatePageUserSheet();
    void invalidateInjectedStyleSheetCache();
    void updateInjectedStyleSheetCache() const;

    WEBCORE_EXPORT void addUserStyleSheet(Ref<StyleSheetContents>&&);

    WEBCORE_EXPORT void addAuthorStyleSheetForTesting(Ref<StyleSheetContents>&&);

#if ENABLE(CONTENT_EXTENSIONS)
    void addDisplayNoneSelector(const String& identifier, const String& selector, uint32_t selectorID);
    void maybeAddContentExtensionSheet(const String& identifier, StyleSheetContents&);
#endif

    void injectPageSpecificUserStyleSheet(const UserStyleSheet&);
    void removePageSpecificUserStyleSheet(const UserStyleSheet&);

    String contentForInjectedStyleSheet(const RefPtr<CSSStyleSheet>&) const;

    void detachFromDocument();

private:
    Ref<Document> protectedDocument() const;

    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;

    RefPtr<CSSStyleSheet> m_pageUserSheet;

    mutable Vector<RefPtr<CSSStyleSheet>> m_injectedUserStyleSheets;
    mutable Vector<RefPtr<CSSStyleSheet>> m_injectedAuthorStyleSheets;
    mutable UncheckedKeyHashMap<RefPtr<CSSStyleSheet>, String> m_injectedStyleSheetToSource;
    mutable bool m_injectedStyleSheetCacheValid { false };

    Vector<RefPtr<CSSStyleSheet>> m_userStyleSheets;
    Vector<RefPtr<CSSStyleSheet>> m_authorStyleSheetsForTesting;
    Vector<UserStyleSheet> m_pageSpecificStyleSheets;

#if ENABLE(CONTENT_EXTENSIONS)
    MemoryCompactRobinHoodHashMap<String, RefPtr<CSSStyleSheet>> m_contentExtensionSheets;
    MemoryCompactRobinHoodHashMap<String, RefPtr<ContentExtensions::ContentExtensionStyleSheet>> m_contentExtensionSelectorSheets;
#endif
};

} // namespace WebCore
