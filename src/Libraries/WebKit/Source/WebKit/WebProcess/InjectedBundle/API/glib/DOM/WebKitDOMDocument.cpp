/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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
#include "WebKitDOMDocument.h"

#include "DOMObjectCache.h"
#include "WebKitDOMDocumentPrivate.h"
#include "WebKitDOMNodePrivate.h"
#include "WebKitDOMPrivate.h"

#if PLATFORM(GTK)
#include "WebKitDOMEventTarget.h"
#endif

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

namespace WebKit {

WebKitDOMDocument* kit(WebCore::Document* obj)
{
    return WEBKIT_DOM_DOCUMENT(kit(static_cast<WebCore::Node*>(obj)));
}

WebCore::Document* core(WebKitDOMDocument* document)
{
    return document ? static_cast<WebCore::Document*>(webkitDOMNodeGetCoreObject(WEBKIT_DOM_NODE(document))) : nullptr;
}

WebKitDOMDocument* wrapDocument(WebCore::Document* coreObject)
{
    ASSERT(coreObject);
#if PLATFORM(GTK)
    return WEBKIT_DOM_DOCUMENT(g_object_new(WEBKIT_DOM_TYPE_DOCUMENT, "core-object", coreObject, nullptr));
#else
    auto* document = WEBKIT_DOM_DOCUMENT(g_object_new(WEBKIT_DOM_TYPE_DOCUMENT, nullptr));
    webkitDOMNodeSetCoreObject(WEBKIT_DOM_NODE(document), static_cast<WebCore::Node*>(coreObject));
    return document;
#endif
}

} // namespace WebKit

#if PLATFORM(GTK)
G_DEFINE_TYPE_WITH_CODE(WebKitDOMDocument, webkit_dom_document, WEBKIT_DOM_TYPE_NODE, G_IMPLEMENT_INTERFACE(WEBKIT_DOM_TYPE_EVENT_TARGET, webkitDOMDocumentDOMEventTargetInit))
#else
G_DEFINE_TYPE(WebKitDOMDocument, webkit_dom_document, WEBKIT_DOM_TYPE_NODE)
#endif

static void webkit_dom_document_class_init(WebKitDOMDocumentClass* documentClass)
{
#if PLATFORM(GTK)
    GObjectClass* gobjectClass = G_OBJECT_CLASS(documentClass);
    webkitDOMDocumentInstallProperties(gobjectClass);
#endif
}

static void webkit_dom_document_init(WebKitDOMDocument*)
{
}

G_GNUC_END_IGNORE_DEPRECATIONS;
