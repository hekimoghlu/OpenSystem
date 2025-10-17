/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#include "WebKitDOMHTMLAppletElement.h"

#include <WebCore/CSSImportRule.h>
#include "DOMObjectCache.h"
#include <WebCore/DOMException.h>
#include <WebCore/Document.h>
#include "GObjectEventListener.h"
#include <WebCore/HTMLNames.h>
#include <WebCore/JSExecState.h>
#include "WebKitDOMEventPrivate.h"
#include "WebKitDOMEventTarget.h"
#include "WebKitDOMNodePrivate.h"
#include "WebKitDOMPrivate.h"
#include "ConvertToUTF8String.h"
#include <wtf/GetPtr.h>
#include <wtf/RefPtr.h>

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

static gboolean webkit_dom_html_applet_element_dispatch_event(WebKitDOMEventTarget*, WebKitDOMEvent*, GError**)
{
    return false;
}

static gboolean webkit_dom_html_applet_element_add_event_listener(WebKitDOMEventTarget*, const char*, GClosure*, gboolean)
{
    return false;
}

static gboolean webkit_dom_html_applet_element_remove_event_listener(WebKitDOMEventTarget*, const char*, GClosure*, gboolean)
{
    return false;
}

static void webkit_dom_html_applet_element_dom_event_target_init(WebKitDOMEventTargetIface* iface)
{
    iface->dispatch_event = webkit_dom_html_applet_element_dispatch_event;
    iface->add_event_listener = webkit_dom_html_applet_element_add_event_listener;
    iface->remove_event_listener = webkit_dom_html_applet_element_remove_event_listener;
}

G_DEFINE_TYPE_WITH_CODE(WebKitDOMHTMLAppletElement, webkit_dom_html_applet_element, WEBKIT_DOM_TYPE_HTML_ELEMENT, G_IMPLEMENT_INTERFACE(WEBKIT_DOM_TYPE_EVENT_TARGET, webkit_dom_html_applet_element_dom_event_target_init))

enum {
    DOM_HTML_APPLET_ELEMENT_PROP_0,
    DOM_HTML_APPLET_ELEMENT_PROP_ALIGN,
    DOM_HTML_APPLET_ELEMENT_PROP_ALT,
    DOM_HTML_APPLET_ELEMENT_PROP_ARCHIVE,
    DOM_HTML_APPLET_ELEMENT_PROP_CODE,
    DOM_HTML_APPLET_ELEMENT_PROP_CODE_BASE,
    DOM_HTML_APPLET_ELEMENT_PROP_HEIGHT,
    DOM_HTML_APPLET_ELEMENT_PROP_HSPACE,
    DOM_HTML_APPLET_ELEMENT_PROP_NAME,
    DOM_HTML_APPLET_ELEMENT_PROP_OBJECT,
    DOM_HTML_APPLET_ELEMENT_PROP_VSPACE,
    DOM_HTML_APPLET_ELEMENT_PROP_WIDTH,
};

static void webkit_dom_html_applet_element_set_property(GObject* object, guint propertyId, const GValue* value, GParamSpec* pspec)
{
    WebKitDOMHTMLAppletElement* self = WEBKIT_DOM_HTML_APPLET_ELEMENT(object);

    switch (propertyId) {
    case DOM_HTML_APPLET_ELEMENT_PROP_ALIGN:
        webkit_dom_html_applet_element_set_align(self, g_value_get_string(value));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_ALT:
        webkit_dom_html_applet_element_set_alt(self, g_value_get_string(value));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_ARCHIVE:
        webkit_dom_html_applet_element_set_archive(self, g_value_get_string(value));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_CODE:
        webkit_dom_html_applet_element_set_code(self, g_value_get_string(value));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_CODE_BASE:
        webkit_dom_html_applet_element_set_code_base(self, g_value_get_string(value));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_HEIGHT:
        webkit_dom_html_applet_element_set_height(self, g_value_get_string(value));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_HSPACE:
        webkit_dom_html_applet_element_set_hspace(self, g_value_get_long(value));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_NAME:
        webkit_dom_html_applet_element_set_name(self, g_value_get_string(value));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_OBJECT:
        webkit_dom_html_applet_element_set_object(self, g_value_get_string(value));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_VSPACE:
        webkit_dom_html_applet_element_set_vspace(self, g_value_get_long(value));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_WIDTH:
        webkit_dom_html_applet_element_set_width(self, g_value_get_string(value));
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, propertyId, pspec);
        break;
    }
}

static void webkit_dom_html_applet_element_get_property(GObject* object, guint propertyId, GValue* value, GParamSpec* pspec)
{
    WebKitDOMHTMLAppletElement* self = WEBKIT_DOM_HTML_APPLET_ELEMENT(object);

    switch (propertyId) {
    case DOM_HTML_APPLET_ELEMENT_PROP_ALIGN:
        g_value_take_string(value, webkit_dom_html_applet_element_get_align(self));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_ALT:
        g_value_take_string(value, webkit_dom_html_applet_element_get_alt(self));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_ARCHIVE:
        g_value_take_string(value, webkit_dom_html_applet_element_get_archive(self));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_CODE:
        g_value_take_string(value, webkit_dom_html_applet_element_get_code(self));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_CODE_BASE:
        g_value_take_string(value, webkit_dom_html_applet_element_get_code_base(self));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_HEIGHT:
        g_value_take_string(value, webkit_dom_html_applet_element_get_height(self));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_HSPACE:
        g_value_set_long(value, webkit_dom_html_applet_element_get_hspace(self));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_NAME:
        g_value_take_string(value, webkit_dom_html_applet_element_get_name(self));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_OBJECT:
        g_value_take_string(value, webkit_dom_html_applet_element_get_object(self));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_VSPACE:
        g_value_set_long(value, webkit_dom_html_applet_element_get_vspace(self));
        break;
    case DOM_HTML_APPLET_ELEMENT_PROP_WIDTH:
        g_value_take_string(value, webkit_dom_html_applet_element_get_width(self));
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, propertyId, pspec);
        break;
    }
}

static void webkit_dom_html_applet_element_class_init(WebKitDOMHTMLAppletElementClass* requestClass)
{
    GObjectClass* gobjectClass = G_OBJECT_CLASS(requestClass);
    gobjectClass->set_property = webkit_dom_html_applet_element_set_property;
    gobjectClass->get_property = webkit_dom_html_applet_element_get_property;

    g_object_class_install_property(
        gobjectClass,
        DOM_HTML_APPLET_ELEMENT_PROP_ALIGN,
        g_param_spec_string(
            "align",
            "HTMLAppletElement:align",
            "read-write gchar* HTMLAppletElement:align",
            "",
            WEBKIT_PARAM_READWRITE));

    g_object_class_install_property(
        gobjectClass,
        DOM_HTML_APPLET_ELEMENT_PROP_ALT,
        g_param_spec_string(
            "alt",
            "HTMLAppletElement:alt",
            "read-write gchar* HTMLAppletElement:alt",
            "",
            WEBKIT_PARAM_READWRITE));

    g_object_class_install_property(
        gobjectClass,
        DOM_HTML_APPLET_ELEMENT_PROP_ARCHIVE,
        g_param_spec_string(
            "archive",
            "HTMLAppletElement:archive",
            "read-write gchar* HTMLAppletElement:archive",
            "",
            WEBKIT_PARAM_READWRITE));

    g_object_class_install_property(
        gobjectClass,
        DOM_HTML_APPLET_ELEMENT_PROP_CODE,
        g_param_spec_string(
            "code",
            "HTMLAppletElement:code",
            "read-write gchar* HTMLAppletElement:code",
            "",
            WEBKIT_PARAM_READWRITE));

    g_object_class_install_property(
        gobjectClass,
        DOM_HTML_APPLET_ELEMENT_PROP_CODE_BASE,
        g_param_spec_string(
            "code-base",
            "HTMLAppletElement:code-base",
            "read-write gchar* HTMLAppletElement:code-base",
            "",
            WEBKIT_PARAM_READWRITE));

    g_object_class_install_property(
        gobjectClass,
        DOM_HTML_APPLET_ELEMENT_PROP_HEIGHT,
        g_param_spec_string(
            "height",
            "HTMLAppletElement:height",
            "read-write gchar* HTMLAppletElement:height",
            "",
            WEBKIT_PARAM_READWRITE));

    g_object_class_install_property(
        gobjectClass,
        DOM_HTML_APPLET_ELEMENT_PROP_HSPACE,
        g_param_spec_long(
            "hspace",
            "HTMLAppletElement:hspace",
            "read-write glong HTMLAppletElement:hspace",
            G_MINLONG, G_MAXLONG, 0,
            WEBKIT_PARAM_READWRITE));

    g_object_class_install_property(
        gobjectClass,
        DOM_HTML_APPLET_ELEMENT_PROP_NAME,
        g_param_spec_string(
            "name",
            "HTMLAppletElement:name",
            "read-write gchar* HTMLAppletElement:name",
            "",
            WEBKIT_PARAM_READWRITE));

    g_object_class_install_property(
        gobjectClass,
        DOM_HTML_APPLET_ELEMENT_PROP_OBJECT,
        g_param_spec_string(
            "object",
            "HTMLAppletElement:object",
            "read-write gchar* HTMLAppletElement:object",
            "",
            WEBKIT_PARAM_READWRITE));

    g_object_class_install_property(
        gobjectClass,
        DOM_HTML_APPLET_ELEMENT_PROP_VSPACE,
        g_param_spec_long(
            "vspace",
            "HTMLAppletElement:vspace",
            "read-write glong HTMLAppletElement:vspace",
            G_MINLONG, G_MAXLONG, 0,
            WEBKIT_PARAM_READWRITE));

    g_object_class_install_property(
        gobjectClass,
        DOM_HTML_APPLET_ELEMENT_PROP_WIDTH,
        g_param_spec_string(
            "width",
            "HTMLAppletElement:width",
            "read-write gchar* HTMLAppletElement:width",
            "",
            WEBKIT_PARAM_READWRITE));

}

static void webkit_dom_html_applet_element_init(WebKitDOMHTMLAppletElement* request)
{
    UNUSED_PARAM(request);
}

gchar* webkit_dom_html_applet_element_get_align(WebKitDOMHTMLAppletElement*)
{
    return nullptr;
}

void webkit_dom_html_applet_element_set_align(WebKitDOMHTMLAppletElement*, const gchar*)
{
}

gchar* webkit_dom_html_applet_element_get_alt(WebKitDOMHTMLAppletElement*)
{
    return nullptr;
}

void webkit_dom_html_applet_element_set_alt(WebKitDOMHTMLAppletElement*, const gchar*)
{
}

gchar* webkit_dom_html_applet_element_get_archive(WebKitDOMHTMLAppletElement*)
{
    return nullptr;
}

void webkit_dom_html_applet_element_set_archive(WebKitDOMHTMLAppletElement*, const gchar*)
{
}

gchar* webkit_dom_html_applet_element_get_code(WebKitDOMHTMLAppletElement*)
{
    return nullptr;
}

void webkit_dom_html_applet_element_set_code(WebKitDOMHTMLAppletElement*, const gchar*)
{
}

gchar* webkit_dom_html_applet_element_get_code_base(WebKitDOMHTMLAppletElement*)
{
    return nullptr;
}

void webkit_dom_html_applet_element_set_code_base(WebKitDOMHTMLAppletElement*, const gchar*)
{
}

gchar* webkit_dom_html_applet_element_get_height(WebKitDOMHTMLAppletElement*)
{
    return nullptr;
}

void webkit_dom_html_applet_element_set_height(WebKitDOMHTMLAppletElement*, const gchar*)
{
}

glong webkit_dom_html_applet_element_get_hspace(WebKitDOMHTMLAppletElement*)
{
    return 0;
}

void webkit_dom_html_applet_element_set_hspace(WebKitDOMHTMLAppletElement*, glong)
{
}

gchar* webkit_dom_html_applet_element_get_name(WebKitDOMHTMLAppletElement*)
{
    return nullptr;
}

void webkit_dom_html_applet_element_set_name(WebKitDOMHTMLAppletElement*, const gchar*)
{
}

gchar* webkit_dom_html_applet_element_get_object(WebKitDOMHTMLAppletElement*)
{
    return nullptr;
}

void webkit_dom_html_applet_element_set_object(WebKitDOMHTMLAppletElement*, const gchar*)
{
}

glong webkit_dom_html_applet_element_get_vspace(WebKitDOMHTMLAppletElement*)
{
    return 0;
}

void webkit_dom_html_applet_element_set_vspace(WebKitDOMHTMLAppletElement*, glong)
{
}

gchar* webkit_dom_html_applet_element_get_width(WebKitDOMHTMLAppletElement*)
{
    return nullptr;
}

void webkit_dom_html_applet_element_set_width(WebKitDOMHTMLAppletElement*, const gchar*)
{
}

G_GNUC_END_IGNORE_DEPRECATIONS;
