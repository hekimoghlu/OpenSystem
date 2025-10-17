/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#include "CSSStyleSheetObservableArray.h"

#include "Document.h"
#include "JSCSSStyleSheet.h"
#include "JSDOMConvert.h"
#include "ShadowRoot.h"
#include "TreeScope.h"

namespace WebCore {

Ref<CSSStyleSheetObservableArray> CSSStyleSheetObservableArray::create(ContainerNode& treeScope)
{
    return adoptRef(*new CSSStyleSheetObservableArray(treeScope));
}

CSSStyleSheetObservableArray::CSSStyleSheetObservableArray(ContainerNode& treeScope)
    : m_treeScope(treeScope)
{
    ASSERT(is<Document>(treeScope) || is<ShadowRoot>(treeScope));
}

bool CSSStyleSheetObservableArray::setValueAt(JSC::JSGlobalObject* lexicalGlobalObject, unsigned index, JSC::JSValue value)
{
    auto& vm = lexicalGlobalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    RELEASE_ASSERT(index <= m_sheets.size());

    auto sheetConversionResult = convert<IDLInterface<CSSStyleSheet>>(*lexicalGlobalObject, value);
    if (UNLIKELY(sheetConversionResult.hasException(scope)))
        return false;

    if (auto exception = shouldThrowWhenAddingSheet(*sheetConversionResult.returnValue())) {
        throwException(lexicalGlobalObject, scope, createDOMException(*lexicalGlobalObject, WTFMove(*exception)));
        return false;
    }

    if (index == m_sheets.size())
        m_sheets.append(*sheetConversionResult.returnValue());
    else
        m_sheets[index] = *sheetConversionResult.returnValue();

    didAddSheet(*sheetConversionResult.releaseReturnValue());
    return true;
}

void CSSStyleSheetObservableArray::removeLast()
{
    RELEASE_ASSERT(!m_sheets.isEmpty());
    auto sheet = m_sheets.takeLast();
    willRemoveSheet(sheet);
}

void CSSStyleSheetObservableArray::shrinkTo(unsigned length)
{
    RELEASE_ASSERT(length <= m_sheets.size());
    m_sheets.shrink(length);
}

JSC::JSValue CSSStyleSheetObservableArray::valueAt(JSC::JSGlobalObject* lexicalGlobalObject, unsigned index) const
{
    if (index >= m_sheets.size())
        return JSC::jsUndefined();
    return toJS(lexicalGlobalObject, JSC::jsCast<JSDOMGlobalObject*>(lexicalGlobalObject), m_sheets[index]);
}

ExceptionOr<void> CSSStyleSheetObservableArray::setSheets(Vector<Ref<CSSStyleSheet>>&& sheets)
{
    for (auto& sheet : sheets) {
        if (auto exception = shouldThrowWhenAddingSheet(sheet))
            return WTFMove(*exception);
    }

    for (auto& sheet : m_sheets)
        willRemoveSheet(sheet);
    m_sheets = WTFMove(sheets);
    for (auto& sheet : m_sheets)
        didAddSheet(sheet);

    return { };
}

std::optional<Exception> CSSStyleSheetObservableArray::shouldThrowWhenAddingSheet(const CSSStyleSheet& sheet) const
{
    if (!sheet.wasConstructedByJS())
        return Exception { ExceptionCode::NotAllowedError, "Sheet needs to be constructed by JavaScript"_s };
    auto* treeScope = this->treeScope();
    if (!treeScope || sheet.constructorDocument() != &treeScope->documentScope())
        return Exception { ExceptionCode::NotAllowedError, "Sheet constructor document doesn't match"_s };
    return std::nullopt;
}

TreeScope* CSSStyleSheetObservableArray::treeScope() const
{
    if (!m_treeScope)
        return nullptr;
    if (auto* shadowRoot = dynamicDowncast<ShadowRoot>(*m_treeScope))
        return shadowRoot;
    return &downcast<Document>(*m_treeScope);
}

void CSSStyleSheetObservableArray::didAddSheet(CSSStyleSheet& sheet)
{
    if (m_treeScope)
        sheet.addAdoptingTreeScope(*m_treeScope);
}

void CSSStyleSheetObservableArray::willRemoveSheet(CSSStyleSheet& sheet)
{
    if (m_treeScope)
        sheet.removeAdoptingTreeScope(*m_treeScope);
}

} // namespace WebCore
