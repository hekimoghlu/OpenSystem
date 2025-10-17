/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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

#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class HTMLStyleElement;
class Node;
class ShadowRoot;
class StyleSheet;
class CSSStyleSheet;
class WeakPtrImplWithEventTargetData;

class StyleSheetList final : public RefCounted<StyleSheetList> {
public:
    static Ref<StyleSheetList> create(Document& document) { return adoptRef(*new StyleSheetList(document)); }
    static Ref<StyleSheetList> create(ShadowRoot& shadowRoot) { return adoptRef(*new StyleSheetList(shadowRoot)); }
    WEBCORE_EXPORT ~StyleSheetList();

    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    WEBCORE_EXPORT unsigned length() const;
    WEBCORE_EXPORT StyleSheet* item(unsigned index);

    CSSStyleSheet* namedItem(const AtomString&) const;
    bool isSupportedPropertyName(const AtomString&) const;
    Vector<AtomString> supportedPropertyNames();

    Node* ownerNode() const;

    void detach();

private:
    StyleSheetList(Document&);
    StyleSheetList(ShadowRoot&);
    const Vector<RefPtr<StyleSheet>>& styleSheets() const;

    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
    ShadowRoot* m_shadowRoot { nullptr };
    Vector<RefPtr<StyleSheet>> m_detachedStyleSheets;
};

} // namespace WebCore
