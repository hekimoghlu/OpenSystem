/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#include "config.h"
#include "StyleSheetList.h"

#include "CSSStyleSheet.h"
#include "Document.h"
#include "HTMLNames.h"
#include "HTMLStyleElement.h"
#include "ShadowRoot.h"
#include "StyleScope.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

using namespace HTMLNames;

StyleSheetList::StyleSheetList(Document& document)
    : m_document(document)
{
}

StyleSheetList::StyleSheetList(ShadowRoot& shadowRoot)
    : m_shadowRoot(&shadowRoot)
{
}

StyleSheetList::~StyleSheetList() = default;

inline const Vector<RefPtr<StyleSheet>>& StyleSheetList::styleSheets() const
{
    if (m_document)
        return m_document->styleScope().styleSheetsForStyleSheetList();
    if (m_shadowRoot)
        return m_shadowRoot->styleScope().styleSheetsForStyleSheetList();
    return m_detachedStyleSheets;
}

Node* StyleSheetList::ownerNode() const
{
    if (m_document)
        return m_document.get();
    return m_shadowRoot;
}

void StyleSheetList::detach()
{
    if (m_document) {
        ASSERT(!m_shadowRoot);
        m_detachedStyleSheets = m_document->styleScope().styleSheetsForStyleSheetList();
        m_document = nullptr;
    } else if (m_shadowRoot) {
        ASSERT(!m_document);
        m_detachedStyleSheets = m_shadowRoot->styleScope().styleSheetsForStyleSheetList();
        m_shadowRoot = nullptr;
    } else
        ASSERT_NOT_REACHED();
}

unsigned StyleSheetList::length() const
{
    return styleSheets().size();
}

StyleSheet* StyleSheetList::item(unsigned index)
{
    const Vector<RefPtr<StyleSheet>>& sheets = styleSheets();
    return index < sheets.size() ? sheets[index].get() : 0;
}

CSSStyleSheet* StyleSheetList::namedItem(const AtomString& name) const
{
    // Support the named getter on document for backwards compatibility.
    if (!m_document)
        return nullptr;

    // IE also supports retrieving a stylesheet by name, using the name/id of the <style> tag
    // (this is consistent with all the other collections)
    // ### Bad implementation because returns a single element (are IDs always unique?)
    // and doesn't look for name attribute.
    // But unicity of stylesheet ids is good practice anyway ;)
    if (RefPtr element = dynamicDowncast<HTMLStyleElement>(m_document->getElementById(name)))
        return element->sheet();
    return nullptr;
}

bool StyleSheetList::isSupportedPropertyName(const AtomString& name) const
{
    return namedItem(name);
}

Vector<AtomString> StyleSheetList::supportedPropertyNames()
{
    // FIXME: Should be implemented.
    return Vector<AtomString>();
}

} // namespace WebCore
