/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#include "WebKitDOMObject.h"

#if PLATFORM(GTK)
enum {
    PROP_0,
    PROP_CORE_OBJECT
};
#endif

G_GNUC_BEGIN_IGNORE_DEPRECATIONS;

G_DEFINE_TYPE(WebKitDOMObject, webkit_dom_object, G_TYPE_OBJECT)

static void webkit_dom_object_init(WebKitDOMObject*)
{
}

#if PLATFORM(GTK)
static void webkitDOMObjectSetProperty(GObject* object, guint propertyId, const GValue* value, GParamSpec* pspec)
{
    switch (propertyId) {
    case PROP_CORE_OBJECT:
        WEBKIT_DOM_OBJECT(object)->coreObject = g_value_get_pointer(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, propertyId, pspec);
        break;
    }
}
#endif

static void webkit_dom_object_class_init(WebKitDOMObjectClass* klass)
{
#if PLATFORM(GTK)
    GObjectClass* gobjectClass = G_OBJECT_CLASS(klass);
    gobjectClass->set_property = webkitDOMObjectSetProperty;

    g_object_class_install_property(
        gobjectClass,
        PROP_CORE_OBJECT,
        g_param_spec_pointer(
            "core-object",
            nullptr, nullptr,
            static_cast<GParamFlags>(G_PARAM_WRITABLE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_NAME | G_PARAM_STATIC_NICK | G_PARAM_STATIC_BLURB)));
#endif
}

G_GNUC_END_IGNORE_DEPRECATIONS;
