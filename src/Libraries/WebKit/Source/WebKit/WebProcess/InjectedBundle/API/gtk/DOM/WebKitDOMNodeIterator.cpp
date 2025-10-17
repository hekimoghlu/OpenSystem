/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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
#include "WebKitDOMNodeIterator.h"

#include <WebCore/CSSImportRule.h>
#include "DOMObjectCache.h"
#include <WebCore/Document.h>
#include <WebCore/ExceptionCode.h>
#include <WebCore/JSExecState.h>
#include "WebKitDOMNodeFilterPrivate.h"
#include "WebKitDOMNodeIteratorPrivate.h"
#include "WebKitDOMNodePrivate.h"
#include "WebKitDOMPrivate.h"
#include "ConvertToUTF8String.h"
#include <wtf/GetPtr.h>
#include <wtf/RefPtr.h>

#define WEBKIT_DOM_NODE_ITERATOR_GET_PRIVATE(obj) G_TYPE_INSTANCE_GET_PRIVATE(obj, WEBKIT_DOM_TYPE_NODE_ITERATOR, WebKitDOMNodeIteratorPrivate)

typedef struct _WebKitDOMNodeIteratorPrivate {
    RefPtr<WebCore::NodeIterator> coreObject;
} WebKitDOMNodeIteratorPrivate;

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

namespace WebKit {

WebKitDOMNodeIterator* kit(WebCore::NodeIterator* obj)
{
    if (!obj)
        return 0;

    if (gpointer ret = DOMObjectCache::get(obj))
        return WEBKIT_DOM_NODE_ITERATOR(ret);

    return wrapNodeIterator(obj);
}

WebCore::NodeIterator* core(WebKitDOMNodeIterator* request)
{
    return request ? static_cast<WebCore::NodeIterator*>(WEBKIT_DOM_OBJECT(request)->coreObject) : 0;
}

WebKitDOMNodeIterator* wrapNodeIterator(WebCore::NodeIterator* coreObject)
{
    ASSERT(coreObject);
    return WEBKIT_DOM_NODE_ITERATOR(g_object_new(WEBKIT_DOM_TYPE_NODE_ITERATOR, "core-object", coreObject, nullptr));
}

} // namespace WebKit

G_DEFINE_TYPE(WebKitDOMNodeIterator, webkit_dom_node_iterator, WEBKIT_DOM_TYPE_OBJECT)

enum {
    DOM_NODE_ITERATOR_PROP_0,
    DOM_NODE_ITERATOR_PROP_ROOT,
    DOM_NODE_ITERATOR_PROP_WHAT_TO_SHOW,
    DOM_NODE_ITERATOR_PROP_FILTER,
    DOM_NODE_ITERATOR_PROP_REFERENCE_NODE,
    DOM_NODE_ITERATOR_PROP_POINTER_BEFORE_REFERENCE_NODE,
};

static void webkit_dom_node_iterator_finalize(GObject* object)
{
    WebKitDOMNodeIteratorPrivate* priv = WEBKIT_DOM_NODE_ITERATOR_GET_PRIVATE(object);

    WebKit::DOMObjectCache::forget(priv->coreObject.get());

    priv->~WebKitDOMNodeIteratorPrivate();
    G_OBJECT_CLASS(webkit_dom_node_iterator_parent_class)->finalize(object);
}

static void webkit_dom_node_iterator_get_property(GObject* object, guint propertyId, GValue* value, GParamSpec* pspec)
{
    WebKitDOMNodeIterator* self = WEBKIT_DOM_NODE_ITERATOR(object);

    switch (propertyId) {
    case DOM_NODE_ITERATOR_PROP_ROOT:
        g_value_set_object(value, webkit_dom_node_iterator_get_root(self));
        break;
    case DOM_NODE_ITERATOR_PROP_WHAT_TO_SHOW:
        g_value_set_ulong(value, webkit_dom_node_iterator_get_what_to_show(self));
        break;
    case DOM_NODE_ITERATOR_PROP_FILTER:
        g_value_set_object(value, webkit_dom_node_iterator_get_filter(self));
        break;
    case DOM_NODE_ITERATOR_PROP_REFERENCE_NODE:
        g_value_set_object(value, webkit_dom_node_iterator_get_reference_node(self));
        break;
    case DOM_NODE_ITERATOR_PROP_POINTER_BEFORE_REFERENCE_NODE:
        g_value_set_boolean(value, webkit_dom_node_iterator_get_pointer_before_reference_node(self));
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, propertyId, pspec);
        break;
    }
}

static GObject* webkit_dom_node_iterator_constructor(GType type, guint constructPropertiesCount, GObjectConstructParam* constructProperties)
{
    GObject* object = G_OBJECT_CLASS(webkit_dom_node_iterator_parent_class)->constructor(type, constructPropertiesCount, constructProperties);

    WebKitDOMNodeIteratorPrivate* priv = WEBKIT_DOM_NODE_ITERATOR_GET_PRIVATE(object);
    priv->coreObject = static_cast<WebCore::NodeIterator*>(WEBKIT_DOM_OBJECT(object)->coreObject);
    WebKit::DOMObjectCache::put(priv->coreObject.get(), object);

    return object;
}

static void webkit_dom_node_iterator_class_init(WebKitDOMNodeIteratorClass* requestClass)
{
    GObjectClass* gobjectClass = G_OBJECT_CLASS(requestClass);
    g_type_class_add_private(gobjectClass, sizeof(WebKitDOMNodeIteratorPrivate));
    gobjectClass->constructor = webkit_dom_node_iterator_constructor;
    gobjectClass->finalize = webkit_dom_node_iterator_finalize;
    gobjectClass->get_property = webkit_dom_node_iterator_get_property;

    g_object_class_install_property(
        gobjectClass,
        DOM_NODE_ITERATOR_PROP_ROOT,
        g_param_spec_object(
            "root",
            "NodeIterator:root",
            "read-only WebKitDOMNode* NodeIterator:root",
            WEBKIT_DOM_TYPE_NODE,
            WEBKIT_PARAM_READABLE));

    g_object_class_install_property(
        gobjectClass,
        DOM_NODE_ITERATOR_PROP_WHAT_TO_SHOW,
        g_param_spec_ulong(
            "what-to-show",
            "NodeIterator:what-to-show",
            "read-only gulong NodeIterator:what-to-show",
            0, G_MAXULONG, 0,
            WEBKIT_PARAM_READABLE));

    g_object_class_install_property(
        gobjectClass,
        DOM_NODE_ITERATOR_PROP_FILTER,
        g_param_spec_object(
            "filter",
            "NodeIterator:filter",
            "read-only WebKitDOMNodeFilter* NodeIterator:filter",
            WEBKIT_DOM_TYPE_NODE_FILTER,
            WEBKIT_PARAM_READABLE));

    g_object_class_install_property(
        gobjectClass,
        DOM_NODE_ITERATOR_PROP_REFERENCE_NODE,
        g_param_spec_object(
            "reference-node",
            "NodeIterator:reference-node",
            "read-only WebKitDOMNode* NodeIterator:reference-node",
            WEBKIT_DOM_TYPE_NODE,
            WEBKIT_PARAM_READABLE));

    g_object_class_install_property(
        gobjectClass,
        DOM_NODE_ITERATOR_PROP_POINTER_BEFORE_REFERENCE_NODE,
        g_param_spec_boolean(
            "pointer-before-reference-node",
            "NodeIterator:pointer-before-reference-node",
            "read-only gboolean NodeIterator:pointer-before-reference-node",
            FALSE,
            WEBKIT_PARAM_READABLE));

}

static void webkit_dom_node_iterator_init(WebKitDOMNodeIterator* request)
{
    WebKitDOMNodeIteratorPrivate* priv = WEBKIT_DOM_NODE_ITERATOR_GET_PRIVATE(request);
    new (priv) WebKitDOMNodeIteratorPrivate();
}

WebKitDOMNode* webkit_dom_node_iterator_next_node(WebKitDOMNodeIterator* self, GError** error)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_NODE_ITERATOR(self), 0);
    UNUSED_PARAM(error);
    WebCore::NodeIterator* item = WebKit::core(self);

    auto result = item->nextNode();
    if (result.hasException())
        return nullptr;

    RefPtr<WebCore::Node> gobjectResult = WTF::getPtr(result.releaseReturnValue());
    return WebKit::kit(gobjectResult.get());
}

WebKitDOMNode* webkit_dom_node_iterator_previous_node(WebKitDOMNodeIterator* self, GError** error)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_NODE_ITERATOR(self), 0);
    UNUSED_PARAM(error);
    WebCore::NodeIterator* item = WebKit::core(self);

    auto result = item->previousNode();
    if (result.hasException())
        return nullptr;

    RefPtr<WebCore::Node> gobjectResult = WTF::getPtr(result.releaseReturnValue());
    return WebKit::kit(gobjectResult.get());
}

void webkit_dom_node_iterator_detach(WebKitDOMNodeIterator* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_if_fail(WEBKIT_DOM_IS_NODE_ITERATOR(self));
    WebCore::NodeIterator* item = WebKit::core(self);
    item->detach();
}

WebKitDOMNode* webkit_dom_node_iterator_get_root(WebKitDOMNodeIterator* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_NODE_ITERATOR(self), 0);
    WebCore::NodeIterator* item = WebKit::core(self);
    RefPtr<WebCore::Node> gobjectResult = WTF::getPtr(item->root());
    return WebKit::kit(gobjectResult.get());
}

gulong webkit_dom_node_iterator_get_what_to_show(WebKitDOMNodeIterator* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_NODE_ITERATOR(self), 0);
    WebCore::NodeIterator* item = WebKit::core(self);
    gulong result = item->whatToShow();
    return result;
}

WebKitDOMNodeFilter* webkit_dom_node_iterator_get_filter(WebKitDOMNodeIterator* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_NODE_ITERATOR(self), 0);
    WebCore::NodeIterator* item = WebKit::core(self);
    RefPtr<WebCore::NodeFilter> gobjectResult = WTF::getPtr(item->filter());
    return WebKit::kit(gobjectResult.get());
}

WebKitDOMNode* webkit_dom_node_iterator_get_reference_node(WebKitDOMNodeIterator* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_NODE_ITERATOR(self), 0);
    WebCore::NodeIterator* item = WebKit::core(self);
    RefPtr<WebCore::Node> gobjectResult = WTF::getPtr(item->referenceNode());
    return WebKit::kit(gobjectResult.get());
}

gboolean webkit_dom_node_iterator_get_pointer_before_reference_node(WebKitDOMNodeIterator* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_NODE_ITERATOR(self), FALSE);
    WebCore::NodeIterator* item = WebKit::core(self);
    gboolean result = item->pointerBeforeReferenceNode();
    return result;
}

G_GNUC_END_IGNORE_DEPRECATIONS;
