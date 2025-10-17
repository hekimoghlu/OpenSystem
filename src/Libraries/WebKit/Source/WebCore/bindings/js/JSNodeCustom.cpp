/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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
#include "JSNode.h"

#include "Attr.h"
#include "CDATASection.h"
#include "Comment.h"
#include "Document.h"
#include "DocumentFragment.h"
#include "DocumentType.h"
#include "FrameDestructionObserverInlines.h"
#include "HTMLCanvasElement.h"
#include "HTMLElement.h"
#include "HTMLNames.h"
#include "JSAttr.h"
#include "JSCDATASection.h"
#include "JSComment.h"
#include "JSDOMBinding.h"
#include "JSDOMWindowCustom.h"
#include "JSDocument.h"
#include "JSDocumentFragment.h"
#include "JSDocumentType.h"
#include "JSEventListener.h"
#include "JSHTMLElement.h"
#include "JSHTMLElementWrapperFactory.h"
#include "JSMathMLElementWrapperFactory.h"
#include "JSProcessingInstruction.h"
#include "JSSVGElementWrapperFactory.h"
#include "JSShadowRoot.h"
#include "JSText.h"
#include "MathMLElement.h"
#include "Node.h"
#include "ProcessingInstruction.h"
#include "RegisteredEventListener.h"
#include "SVGElement.h"
#include "ShadowRoot.h"
#include "GCReachableRef.h"
#include "Text.h"
#include "WebCoreOpaqueRootInlines.h"

namespace WebCore {

using namespace JSC;
using namespace HTMLNames;

bool JSNodeOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    auto& node = jsCast<JSNode*>(handle.slot()->asCell())->wrapped();
    if (!node.isConnected()) {
        if (GCReachableRefMap::contains(node) || node.isInCustomElementReactionQueue()) {
            if (UNLIKELY(reason))
                *reason = "Node is scheduled to be used in an async script invocation)"_s;
            return true;
        }
    }

    if (UNLIKELY(reason))
        *reason = "Connected node"_s;

    return containsWebCoreOpaqueRoot(visitor, node);
}

template<typename Visitor>
void JSNode::visitAdditionalChildren(Visitor& visitor)
{
    addWebCoreOpaqueRoot(visitor, wrapped());
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSNode);

static ALWAYS_INLINE JSValue createWrapperInline(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, Ref<Node>&& node)
{
    ASSERT(!getCachedWrapper(globalObject->world(), node));
    
    JSDOMObject* wrapper;    
    switch (node->nodeType()) {
        case Node::ELEMENT_NODE:
            if (auto* htmlElement = dynamicDowncast<HTMLElement>(node.get()))
                wrapper = createJSHTMLWrapper(globalObject, *htmlElement);
            else if (auto* svgElement = dynamicDowncast<SVGElement>(node.get()))
                wrapper = createJSSVGWrapper(globalObject, *svgElement);
#if ENABLE(MATHML)
            else if (auto* mathmlElement = dynamicDowncast<MathMLElement>(node.get()))
                wrapper = createJSMathMLWrapper(globalObject, *mathmlElement);
#endif
            else
                wrapper = createWrapper<Element>(globalObject, WTFMove(node));
            break;
        case Node::ATTRIBUTE_NODE:
            wrapper = createWrapper<Attr>(globalObject, WTFMove(node));
            break;
        case Node::TEXT_NODE:
            wrapper = createWrapper<Text>(globalObject, WTFMove(node));
            break;
        case Node::CDATA_SECTION_NODE:
            wrapper = createWrapper<CDATASection>(globalObject, WTFMove(node));
            break;
        case Node::PROCESSING_INSTRUCTION_NODE:
            wrapper = createWrapper<ProcessingInstruction>(globalObject, WTFMove(node));
            break;
        case Node::COMMENT_NODE:
            wrapper = createWrapper<Comment>(globalObject, WTFMove(node));
            break;
        case Node::DOCUMENT_NODE:
            // we don't want to cache the document itself in the per-document dictionary
            return toJS(lexicalGlobalObject, globalObject, uncheckedDowncast<Document>(node.get()));
        case Node::DOCUMENT_TYPE_NODE:
            wrapper = createWrapper<DocumentType>(globalObject, WTFMove(node));
            break;
        case Node::DOCUMENT_FRAGMENT_NODE:
            if (node->isShadowRoot())
                wrapper = createWrapper<ShadowRoot>(globalObject, WTFMove(node));
            else
                wrapper = createWrapper<DocumentFragment>(globalObject, WTFMove(node));
            break;
        default:
            wrapper = createWrapper<Node>(globalObject, WTFMove(node));
    }

    return wrapper;
}

JSValue createWrapper(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, Ref<Node>&& node)
{
    return createWrapperInline(lexicalGlobalObject, globalObject, WTFMove(node));
}
    
JSValue toJSNewlyCreated(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, Ref<Node>&& node)
{
    return createWrapperInline(lexicalGlobalObject, globalObject, WTFMove(node));
}

JSC::JSObject* getOutOfLineCachedWrapper(JSDOMGlobalObject* globalObject, Node& node)
{
    ASSERT(!globalObject->world().isNormal());
    return globalObject->world().wrappers().get(&node);
}

void willCreatePossiblyOrphanedTreeByRemovalSlowCase(Node& root)
{
    auto frame = root.document().frame();
    if (!frame)
        return;

    auto& globalObject = mainWorldGlobalObject(*frame);
    JSLockHolder lock(&globalObject);
    ASSERT(!root.wrapper());
    createWrapper(&globalObject, &globalObject, root);
}

} // namespace WebCore
