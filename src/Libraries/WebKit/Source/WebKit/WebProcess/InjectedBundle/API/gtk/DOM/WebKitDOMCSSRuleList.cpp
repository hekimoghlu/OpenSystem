/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#include "WebKitDOMCSSRuleList.h"

#include <WebCore/CSSImportRule.h>
#include "DOMObjectCache.h"
#include <WebCore/Document.h>
#include <WebCore/JSExecState.h>
#include "WebKitDOMCSSRuleListPrivate.h"
#include "WebKitDOMCSSRulePrivate.h"
#include "WebKitDOMPrivate.h"
#include "ConvertToUTF8String.h"
#include <wtf/GetPtr.h>
#include <wtf/RefPtr.h>

#define WEBKIT_DOM_CSS_RULE_LIST_GET_PRIVATE(obj) G_TYPE_INSTANCE_GET_PRIVATE(obj, WEBKIT_DOM_TYPE_CSS_RULE_LIST, WebKitDOMCSSRuleListPrivate)

typedef struct _WebKitDOMCSSRuleListPrivate {
    RefPtr<WebCore::CSSRuleList> coreObject;
} WebKitDOMCSSRuleListPrivate;

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

namespace WebKit {

WebKitDOMCSSRuleList* kit(WebCore::CSSRuleList* obj)
{
    if (!obj)
        return 0;

    if (gpointer ret = DOMObjectCache::get(obj))
        return WEBKIT_DOM_CSS_RULE_LIST(ret);

    return wrapCSSRuleList(obj);
}

WebCore::CSSRuleList* core(WebKitDOMCSSRuleList* request)
{
    return request ? static_cast<WebCore::CSSRuleList*>(WEBKIT_DOM_OBJECT(request)->coreObject) : 0;
}

WebKitDOMCSSRuleList* wrapCSSRuleList(WebCore::CSSRuleList* coreObject)
{
    ASSERT(coreObject);
    return WEBKIT_DOM_CSS_RULE_LIST(g_object_new(WEBKIT_DOM_TYPE_CSS_RULE_LIST, "core-object", coreObject, nullptr));
}

} // namespace WebKit

G_DEFINE_TYPE(WebKitDOMCSSRuleList, webkit_dom_css_rule_list, WEBKIT_DOM_TYPE_OBJECT)

enum {
    DOM_CSS_RULE_LIST_PROP_0,
    DOM_CSS_RULE_LIST_PROP_LENGTH,
};

static void webkit_dom_css_rule_list_finalize(GObject* object)
{
    WebKitDOMCSSRuleListPrivate* priv = WEBKIT_DOM_CSS_RULE_LIST_GET_PRIVATE(object);

    WebKit::DOMObjectCache::forget(priv->coreObject.get());

    priv->~WebKitDOMCSSRuleListPrivate();
    G_OBJECT_CLASS(webkit_dom_css_rule_list_parent_class)->finalize(object);
}

static void webkit_dom_css_rule_list_get_property(GObject* object, guint propertyId, GValue* value, GParamSpec* pspec)
{
    WebKitDOMCSSRuleList* self = WEBKIT_DOM_CSS_RULE_LIST(object);

    switch (propertyId) {
    case DOM_CSS_RULE_LIST_PROP_LENGTH:
        g_value_set_ulong(value, webkit_dom_css_rule_list_get_length(self));
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, propertyId, pspec);
        break;
    }
}

static GObject* webkit_dom_css_rule_list_constructor(GType type, guint constructPropertiesCount, GObjectConstructParam* constructProperties)
{
    GObject* object = G_OBJECT_CLASS(webkit_dom_css_rule_list_parent_class)->constructor(type, constructPropertiesCount, constructProperties);

    WebKitDOMCSSRuleListPrivate* priv = WEBKIT_DOM_CSS_RULE_LIST_GET_PRIVATE(object);
    priv->coreObject = static_cast<WebCore::CSSRuleList*>(WEBKIT_DOM_OBJECT(object)->coreObject);
    WebKit::DOMObjectCache::put(priv->coreObject.get(), object);

    return object;
}

static void webkit_dom_css_rule_list_class_init(WebKitDOMCSSRuleListClass* requestClass)
{
    GObjectClass* gobjectClass = G_OBJECT_CLASS(requestClass);
    g_type_class_add_private(gobjectClass, sizeof(WebKitDOMCSSRuleListPrivate));
    gobjectClass->constructor = webkit_dom_css_rule_list_constructor;
    gobjectClass->finalize = webkit_dom_css_rule_list_finalize;
    gobjectClass->get_property = webkit_dom_css_rule_list_get_property;

    g_object_class_install_property(
        gobjectClass,
        DOM_CSS_RULE_LIST_PROP_LENGTH,
        g_param_spec_ulong(
            "length",
            "CSSRuleList:length",
            "read-only gulong CSSRuleList:length",
            0, G_MAXULONG, 0,
            WEBKIT_PARAM_READABLE));

}

static void webkit_dom_css_rule_list_init(WebKitDOMCSSRuleList* request)
{
    WebKitDOMCSSRuleListPrivate* priv = WEBKIT_DOM_CSS_RULE_LIST_GET_PRIVATE(request);
    new (priv) WebKitDOMCSSRuleListPrivate();
}

WebKitDOMCSSRule* webkit_dom_css_rule_list_item(WebKitDOMCSSRuleList* self, gulong index)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_CSS_RULE_LIST(self), 0);
    WebCore::CSSRuleList* item = WebKit::core(self);
    RefPtr<WebCore::CSSRule> gobjectResult = WTF::getPtr(item->item(index));
    return WebKit::kit(gobjectResult.get());
}

gulong webkit_dom_css_rule_list_get_length(WebKitDOMCSSRuleList* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_CSS_RULE_LIST(self), 0);
    WebCore::CSSRuleList* item = WebKit::core(self);
    gulong result = item->length();
    return result;
}

G_GNUC_END_IGNORE_DEPRECATIONS;
