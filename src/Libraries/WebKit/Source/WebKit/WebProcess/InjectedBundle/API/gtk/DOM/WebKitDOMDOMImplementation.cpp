/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
#include "WebKitDOMDOMImplementation.h"

#include "ConvertToUTF8String.h"
#include "DOMObjectCache.h"
#include "WebKitDOMCSSStyleSheetPrivate.h"
#include "WebKitDOMDOMImplementationPrivate.h"
#include "WebKitDOMDocumentPrivate.h"
#include "WebKitDOMDocumentTypePrivate.h"
#include "WebKitDOMHTMLDocumentPrivate.h"
#include "WebKitDOMPrivate.h"
#include <WebCore/CSSImportRule.h>
#include <WebCore/DOMException.h>
#include <WebCore/Document.h>
#include <WebCore/JSExecState.h>
#include <wtf/GetPtr.h>
#include <wtf/RefPtr.h>

#define WEBKIT_DOM_DOM_IMPLEMENTATION_GET_PRIVATE(obj) G_TYPE_INSTANCE_GET_PRIVATE(obj, WEBKIT_DOM_TYPE_DOM_IMPLEMENTATION, WebKitDOMDOMImplementationPrivate)

typedef struct _WebKitDOMDOMImplementationPrivate {
    RefPtr<WebCore::DOMImplementation> coreObject;
} WebKitDOMDOMImplementationPrivate;

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

namespace WebKit {

WebKitDOMDOMImplementation* kit(WebCore::DOMImplementation* obj)
{
    if (!obj)
        return 0;

    if (gpointer ret = DOMObjectCache::get(obj))
        return WEBKIT_DOM_DOM_IMPLEMENTATION(ret);

    return wrapDOMImplementation(obj);
}

WebCore::DOMImplementation* core(WebKitDOMDOMImplementation* request)
{
    return request ? static_cast<WebCore::DOMImplementation*>(WEBKIT_DOM_OBJECT(request)->coreObject) : 0;
}

WebKitDOMDOMImplementation* wrapDOMImplementation(WebCore::DOMImplementation* coreObject)
{
    ASSERT(coreObject);
    return WEBKIT_DOM_DOM_IMPLEMENTATION(g_object_new(WEBKIT_DOM_TYPE_DOM_IMPLEMENTATION, "core-object", coreObject, nullptr));
}

} // namespace WebKit

G_DEFINE_TYPE(WebKitDOMDOMImplementation, webkit_dom_dom_implementation, WEBKIT_DOM_TYPE_OBJECT)

static void webkit_dom_dom_implementation_finalize(GObject* object)
{
    WebKitDOMDOMImplementationPrivate* priv = WEBKIT_DOM_DOM_IMPLEMENTATION_GET_PRIVATE(object);

    WebKit::DOMObjectCache::forget(priv->coreObject.get());

    priv->~WebKitDOMDOMImplementationPrivate();
    G_OBJECT_CLASS(webkit_dom_dom_implementation_parent_class)->finalize(object);
}

static GObject* webkit_dom_dom_implementation_constructor(GType type, guint constructPropertiesCount, GObjectConstructParam* constructProperties)
{
    GObject* object = G_OBJECT_CLASS(webkit_dom_dom_implementation_parent_class)->constructor(type, constructPropertiesCount, constructProperties);

    WebKitDOMDOMImplementationPrivate* priv = WEBKIT_DOM_DOM_IMPLEMENTATION_GET_PRIVATE(object);
    priv->coreObject = static_cast<WebCore::DOMImplementation*>(WEBKIT_DOM_OBJECT(object)->coreObject);
    WebKit::DOMObjectCache::put(priv->coreObject.get(), object);

    return object;
}

static void webkit_dom_dom_implementation_class_init(WebKitDOMDOMImplementationClass* requestClass)
{
    GObjectClass* gobjectClass = G_OBJECT_CLASS(requestClass);
    g_type_class_add_private(gobjectClass, sizeof(WebKitDOMDOMImplementationPrivate));
    gobjectClass->constructor = webkit_dom_dom_implementation_constructor;
    gobjectClass->finalize = webkit_dom_dom_implementation_finalize;
}

static void webkit_dom_dom_implementation_init(WebKitDOMDOMImplementation* request)
{
    WebKitDOMDOMImplementationPrivate* priv = WEBKIT_DOM_DOM_IMPLEMENTATION_GET_PRIVATE(request);
    new (priv) WebKitDOMDOMImplementationPrivate();
}

gboolean webkit_dom_dom_implementation_has_feature(WebKitDOMDOMImplementation* self, const gchar* feature, const gchar* version)
{
    g_return_val_if_fail(WEBKIT_DOM_IS_DOM_IMPLEMENTATION(self), FALSE);
    g_return_val_if_fail(feature, FALSE);
    g_return_val_if_fail(version, FALSE);
    return TRUE;
}

WebKitDOMDocumentType* webkit_dom_dom_implementation_create_document_type(WebKitDOMDOMImplementation* self, const gchar* qualifiedName, const gchar* publicId, const gchar* systemId, GError** error)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_DOM_IMPLEMENTATION(self), 0);
    g_return_val_if_fail(qualifiedName, 0);
    g_return_val_if_fail(publicId, 0);
    g_return_val_if_fail(systemId, 0);
    g_return_val_if_fail(!error || !*error, 0);
    WebCore::DOMImplementation* item = WebKit::core(self);
    auto convertedQualifiedName = WTF::AtomString::fromUTF8(qualifiedName);
    auto convertedPublicId = WTF::String::fromUTF8(publicId);
    auto convertedSystemId = WTF::String::fromUTF8(systemId);
    auto result = item->createDocumentType(convertedQualifiedName, convertedPublicId, convertedSystemId);
    if (result.hasException()) {
        auto description = WebCore::DOMException::description(result.releaseException().code());
        g_set_error_literal(error, g_quark_from_string("WEBKIT_DOM"), description.legacyCode, description.name);
        return nullptr;
    }
    return WebKit::kit(result.releaseReturnValue().ptr());
}

WebKitDOMDocument* webkit_dom_dom_implementation_create_document(WebKitDOMDOMImplementation* self, const gchar* namespaceURI, const gchar* qualifiedName, WebKitDOMDocumentType* doctype, GError** error)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_DOM_IMPLEMENTATION(self), 0);
    g_return_val_if_fail(qualifiedName, 0);
    g_return_val_if_fail(!doctype || WEBKIT_DOM_IS_DOCUMENT_TYPE(doctype), 0);
    g_return_val_if_fail(!error || !*error, 0);
    WebCore::DOMImplementation* item = WebKit::core(self);
    auto convertedNamespaceURI = WTF::AtomString::fromUTF8(namespaceURI);
    auto convertedQualifiedName = WTF::AtomString::fromUTF8(qualifiedName);
    WebCore::DocumentType* convertedDoctype = WebKit::core(doctype);
    auto result = item->createDocument(convertedNamespaceURI, convertedQualifiedName, convertedDoctype);
    if (result.hasException()) {
        auto description = WebCore::DOMException::description(result.releaseException().code());
        g_set_error_literal(error, g_quark_from_string("WEBKIT_DOM"), description.legacyCode, description.name);
        return nullptr;
    }
    return WebKit::kit(result.releaseReturnValue().ptr());
}

WebKitDOMCSSStyleSheet* webkit_dom_dom_implementation_create_css_style_sheet(WebKitDOMDOMImplementation* self, const gchar* title, const gchar* media, GError** error)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_DOM_IMPLEMENTATION(self), 0);
    g_return_val_if_fail(title, 0);
    g_return_val_if_fail(media, 0);
    g_return_val_if_fail(!error || !*error, 0);
    WebCore::DOMImplementation* item = WebKit::core(self);
    WTF::String convertedTitle = WTF::String::fromUTF8(title);
    WTF::String convertedMedia = WTF::String::fromUTF8(media);
    RefPtr<WebCore::CSSStyleSheet> gobjectResult = WTF::getPtr(item->createCSSStyleSheet(convertedTitle, convertedMedia));
    return WebKit::kit(gobjectResult.get());
}

WebKitDOMHTMLDocument* webkit_dom_dom_implementation_create_html_document(WebKitDOMDOMImplementation* self, const gchar* title)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_DOM_IMPLEMENTATION(self), 0);
    g_return_val_if_fail(title, 0);
    WebCore::DOMImplementation* item = WebKit::core(self);
    WTF::String convertedTitle = WTF::String::fromUTF8(title);
    RefPtr<WebCore::HTMLDocument> gobjectResult = WTF::getPtr(item->createHTMLDocument(WTFMove(convertedTitle)));
    return WebKit::kit(gobjectResult.get());
}

G_GNUC_END_IGNORE_DEPRECATIONS;
