/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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
#include "WebKitDOMPrivate.h"

#include <WebCore/Blob.h>
#include <WebCore/Element.h>
#include <WebCore/Event.h>
#include <WebCore/EventTarget.h>
#include <WebCore/File.h>
#include <WebCore/HTMLCollectionInlines.h>
#include <WebCore/HTMLElement.h>
#include <WebCore/HTMLNames.h>
#include <WebCore/KeyboardEvent.h>
#include <WebCore/MouseEvent.h>
#include <WebCore/StyleSheet.h>
#include <WebCore/UIEvent.h>
#include "WebKitDOMAttrPrivate.h"
#include "WebKitDOMBlobPrivate.h"
#include "WebKitDOMCDATASectionPrivate.h"
#include "WebKitDOMCSSStyleSheetPrivate.h"
#include "WebKitDOMCommentPrivate.h"
#include "WebKitDOMDOMWindowPrivate.h"
#include "WebKitDOMDocumentFragmentPrivate.h"
#include "WebKitDOMDocumentPrivate.h"
#include "WebKitDOMDocumentTypePrivate.h"
#include "WebKitDOMElementPrivate.h"
#include "WebKitDOMEventPrivate.h"
#include "WebKitDOMEventTargetPrivate.h"
#include "WebKitDOMFilePrivate.h"
#include "WebKitDOMHTMLCollectionPrivate.h"
#include "WebKitDOMHTMLDocumentPrivate.h"
#include "WebKitDOMHTMLOptionsCollectionPrivate.h"
#include "WebKitDOMHTMLPrivate.h"
#include "WebKitDOMKeyboardEventPrivate.h"
#include "WebKitDOMMouseEventPrivate.h"
#include "WebKitDOMNodePrivate.h"
#include "WebKitDOMProcessingInstructionPrivate.h"
#include "WebKitDOMStyleSheetPrivate.h"
#include "WebKitDOMTextPrivate.h"
#include "WebKitDOMUIEventPrivate.h"
#include "WebKitDOMWheelEventPrivate.h"

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

namespace WebKit {

using namespace WebCore;
using namespace WebCore::HTMLNames;

WebKitDOMNode* wrapNodeGtk(Node* node)
{
    ASSERT(node);
    ASSERT(node->nodeType());

    switch (node->nodeType()) {
    case Node::ELEMENT_NODE:
        if (is<HTMLElement>(*node))
            return WEBKIT_DOM_NODE(wrap(downcast<HTMLElement>(node)));
        return nullptr;
    case Node::ATTRIBUTE_NODE:
        return WEBKIT_DOM_NODE(wrapAttr(static_cast<Attr*>(node)));
    case Node::TEXT_NODE:
        return WEBKIT_DOM_NODE(wrapText(downcast<Text>(node)));
    case Node::CDATA_SECTION_NODE:
        return WEBKIT_DOM_NODE(wrapCDATASection(static_cast<CDATASection*>(node)));
    case Node::PROCESSING_INSTRUCTION_NODE:
        return WEBKIT_DOM_NODE(wrapProcessingInstruction(static_cast<ProcessingInstruction*>(node)));
    case Node::COMMENT_NODE:
        return WEBKIT_DOM_NODE(wrapComment(static_cast<Comment*>(node)));
    case Node::DOCUMENT_NODE:
        if (is<HTMLDocument>(*node))
            return WEBKIT_DOM_NODE(wrapHTMLDocument(downcast<HTMLDocument>(node)));
        return nullptr;
    case Node::DOCUMENT_TYPE_NODE:
        return WEBKIT_DOM_NODE(wrapDocumentType(static_cast<DocumentType*>(node)));
    case Node::DOCUMENT_FRAGMENT_NODE:
        return WEBKIT_DOM_NODE(wrapDocumentFragment(static_cast<DocumentFragment*>(node)));
    }

    return wrapNode(node);
}

WebKitDOMEvent* wrap(Event* event)
{
    ASSERT(event);

    if (is<UIEvent>(*event)) {
        if (is<MouseEvent>(*event))
            return WEBKIT_DOM_EVENT(wrapMouseEvent(&downcast<MouseEvent>(*event)));
        if (is<KeyboardEvent>(*event))
            return WEBKIT_DOM_EVENT(wrapKeyboardEvent(&downcast<KeyboardEvent>(*event)));
        if (is<WheelEvent>(*event))
            return WEBKIT_DOM_EVENT(wrapWheelEvent(&downcast<WheelEvent>(*event)));
        return WEBKIT_DOM_EVENT(wrapUIEvent(&downcast<UIEvent>(*event)));
    }

    return wrapEvent(event);
}

WebKitDOMStyleSheet* wrap(StyleSheet* styleSheet)
{
    ASSERT(styleSheet);

    if (is<CSSStyleSheet>(*styleSheet))
        return WEBKIT_DOM_STYLE_SHEET(wrapCSSStyleSheet(downcast<CSSStyleSheet>(styleSheet)));
    return wrapStyleSheet(styleSheet);
}

WebKitDOMHTMLCollection* wrap(HTMLCollection* collection)
{
    ASSERT(collection);

    if (is<HTMLOptionsCollection>(*collection))
        return WEBKIT_DOM_HTML_COLLECTION(wrapHTMLOptionsCollection(downcast<HTMLOptionsCollection>(collection)));
    return wrapHTMLCollection(collection);
}

WebKitDOMEventTarget* wrap(EventTarget* eventTarget)
{
    ASSERT(eventTarget);

    if (is<Node>(*eventTarget))
        return WEBKIT_DOM_EVENT_TARGET(kit(&downcast<Node>(*eventTarget)));
    if (is<LocalDOMWindow>(*eventTarget))
        return WEBKIT_DOM_EVENT_TARGET(kit(&downcast<LocalDOMWindow>(*eventTarget)));
    return nullptr;
}

WebKitDOMBlob* wrap(Blob* blob)
{
    ASSERT(blob);

    if (blob->isFile())
        return WEBKIT_DOM_BLOB(wrapFile(static_cast<File*>(blob)));
    return wrapBlob(blob);
}

} // namespace WebKit
G_GNUC_END_IGNORE_DEPRECATIONS;
