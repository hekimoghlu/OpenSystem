/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#include "JSDocument.h"

#include "FrameDestructionObserverInlines.h"
#include "JSCSSStyleSheet.h"
#include "JSDOMConvert.h"
#include "JSDOMGlobalObjectInlines.h"
#include "JSDOMWindowCustom.h"
#include "JSHTMLDocument.h"
#include "JSXMLDocument.h"
#include "LocalFrame.h"
#include "NodeTraversal.h"
#include "WebCoreOpaqueRootInlines.h"
#include <JavaScriptCore/HeapAnalyzer.h>

namespace WebCore {
using namespace JSC;

static inline JSValue createNewDocumentWrapper(JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, Ref<Document>&& passedDocument)
{
    auto& document = passedDocument.get();
    JSObject* wrapper;
    if (document.isHTMLDocument())
        wrapper = createWrapper<HTMLDocument>(&globalObject, WTFMove(passedDocument));
    else if (document.isXMLDocument())
        wrapper = createWrapper<XMLDocument>(&globalObject, WTFMove(passedDocument));
    else
        wrapper = createWrapper<Document>(&globalObject, WTFMove(passedDocument));

    reportMemoryForDocumentIfFrameless(lexicalGlobalObject, document);

    return wrapper;
}

JSObject* cachedDocumentWrapper(JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, Document& document)
{
    if (auto* wrapper = getCachedWrapper(globalObject.world(), document))
        return wrapper;

    RefPtr window = document.domWindow();
    if (!window)
        return nullptr;

    auto* documentGlobalObject = toJSDOMGlobalObject<JSDOMWindow>(lexicalGlobalObject.vm(), toJS(&lexicalGlobalObject, *window));
    if (!documentGlobalObject)
        return nullptr;

    // Creating a wrapper for domWindow might have created a wrapper for document as well.
    return getCachedWrapper(documentGlobalObject->world(), document);
}

void reportMemoryForDocumentIfFrameless(JSGlobalObject& lexicalGlobalObject, Document& document)
{
    // Make sure the document is kept around by the window object, and works right with the back/forward cache.
    if (document.frame())
        return;

    VM& vm = lexicalGlobalObject.vm();
    size_t memoryCost = 0;
    for (Node* node = &document; node; node = NodeTraversal::next(*node))
        memoryCost += node->approximateMemoryCost();

    // FIXME: Adopt reportExtraMemoryVisited, and switch to reportExtraMemoryAllocated.
    // https://bugs.webkit.org/show_bug.cgi?id=142595
    vm.heap.deprecatedReportExtraMemory(memoryCost);
}

JSValue toJSNewlyCreated(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, Ref<Document>&& document)
{
    return createNewDocumentWrapper(*lexicalGlobalObject, *globalObject, WTFMove(document));
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, Document& document)
{
    if (auto* wrapper = cachedDocumentWrapper(*lexicalGlobalObject, *globalObject, document))
        return wrapper;
    return toJSNewlyCreated(lexicalGlobalObject, globalObject, Ref<Document>(document));
}

void setAdoptedStyleSheetsOnTreeScope(TreeScope& treeScope, JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    auto nativeValue = convert<IDLFrozenArray<IDLInterface<CSSStyleSheet>>>(lexicalGlobalObject, value);
    if (UNLIKELY(nativeValue.hasException(throwScope)))
        return;

    auto result = treeScope.setAdoptedStyleSheets(nativeValue.releaseReturnValue());
    if (UNLIKELY(result.hasException()))
        propagateException(lexicalGlobalObject, throwScope, result.releaseException());
}

void JSDocument::setAdoptedStyleSheets(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
{
    setAdoptedStyleSheetsOnTreeScope(wrapped(), lexicalGlobalObject, value);
}

template<typename Visitor>
void JSDocument::visitAdditionalChildren(Visitor& visitor)
{
    addWebCoreOpaqueRoot(visitor, static_cast<ScriptExecutionContext&>(wrapped()));
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSDocument);

void JSDocument::analyzeHeap(JSCell* cell, HeapAnalyzer& analyzer)
{
    Base::analyzeHeap(cell, analyzer);
    auto* thisObject = jsCast<JSDocument*>(cell);
    analyzer.setLabelForCell(cell, thisObject->wrapped().url().string());
}

} // namespace WebCore
