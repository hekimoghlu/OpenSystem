/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
#include "WebKitDOMFile.h"

#include <WebCore/CSSImportRule.h>
#include "DOMObjectCache.h"
#include <WebCore/Document.h>
#include <WebCore/ExceptionCode.h>
#include <WebCore/JSExecState.h>
#include "WebKitDOMBlobPrivate.h"
#include "WebKitDOMFilePrivate.h"
#include "WebKitDOMPrivate.h"
#include "ConvertToUTF8String.h"
#include <wtf/GetPtr.h>
#include <wtf/RefPtr.h>

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

namespace WebKit {

WebKitDOMFile* kit(WebCore::File* obj)
{
    return WEBKIT_DOM_FILE(kit(static_cast<WebCore::Blob*>(obj)));
}

WebCore::File* core(WebKitDOMFile* request)
{
    return request ? static_cast<WebCore::File*>(WEBKIT_DOM_OBJECT(request)->coreObject) : 0;
}

WebKitDOMFile* wrapFile(WebCore::File* coreObject)
{
    ASSERT(coreObject);
    return WEBKIT_DOM_FILE(g_object_new(WEBKIT_DOM_TYPE_FILE, "core-object", coreObject, nullptr));
}

} // namespace WebKit

G_DEFINE_TYPE(WebKitDOMFile, webkit_dom_file, WEBKIT_DOM_TYPE_BLOB)

enum {
    DOM_FILE_PROP_0,
    DOM_FILE_PROP_NAME,
};

static void webkit_dom_file_get_property(GObject* object, guint propertyId, GValue* value, GParamSpec* pspec)
{
    WebKitDOMFile* self = WEBKIT_DOM_FILE(object);

    switch (propertyId) {
    case DOM_FILE_PROP_NAME:
        g_value_take_string(value, webkit_dom_file_get_name(self));
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, propertyId, pspec);
        break;
    }
}

static void webkit_dom_file_class_init(WebKitDOMFileClass* requestClass)
{
    GObjectClass* gobjectClass = G_OBJECT_CLASS(requestClass);
    gobjectClass->get_property = webkit_dom_file_get_property;

    g_object_class_install_property(
        gobjectClass,
        DOM_FILE_PROP_NAME,
        g_param_spec_string(
            "name",
            "File:name",
            "read-only gchar* File:name",
            "",
            WEBKIT_PARAM_READABLE));
}

static void webkit_dom_file_init(WebKitDOMFile* request)
{
    UNUSED_PARAM(request);
}

gchar* webkit_dom_file_get_name(WebKitDOMFile* self)
{
    WebCore::JSMainThreadNullState state;
    g_return_val_if_fail(WEBKIT_DOM_IS_FILE(self), 0);
    WebCore::File* item = WebKit::core(self);
    gchar* result = convertToUTF8String(item->name());
    return result;
}
G_GNUC_END_IGNORE_DEPRECATIONS;
