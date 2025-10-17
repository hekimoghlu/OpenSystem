/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
#include "AccessibilityObjectAtspi.h"

#if USE(ATSPI)

#include "Document.h"
#include "DocumentInlines.h"
#include "DocumentType.h"
#include <gio/gio.h>
#include <wtf/HashMap.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

GDBusInterfaceVTable AccessibilityObjectAtspi::s_documentFunctions = {
    // method_call
    [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar* methodName, GVariant* parameters, GDBusMethodInvocation* invocation, gpointer userData) {
        auto atspiObject = Ref { *static_cast<AccessibilityObjectAtspi*>(userData) };
        atspiObject->updateBackingStore();

        if (!g_strcmp0(methodName, "GetAttributeValue")) {
            const char* name;
            g_variant_get(parameters, "(&s)", &name);
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(s)", atspiObject->documentAttribute(String::fromUTF8(name)).utf8().data()));
        } else if (!g_strcmp0(methodName, "GetAttributes")) {
            GVariantBuilder builder = G_VARIANT_BUILDER_INIT(G_VARIANT_TYPE("(a{ss})"));
            g_variant_builder_open(&builder, G_VARIANT_TYPE("a{ss}"));
            auto attributes = atspiObject->documentAttributes();
            for (const auto& it : attributes)
                g_variant_builder_add(&builder, "{ss}", it.key.utf8().data(), it.value.utf8().data());
            g_variant_builder_close(&builder);
            g_dbus_method_invocation_return_value(invocation, g_variant_builder_end(&builder));
        } else if (!g_strcmp0(methodName, "GetLocale"))
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(s)", atspiObject->documentLocale().utf8().data()));
    },
    // get_property
    [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar* propertyName, GError** error, gpointer userData) -> GVariant* {
        auto atspiObject = Ref { *static_cast<AccessibilityObjectAtspi*>(userData) };
        atspiObject->updateBackingStore();

        if (!g_strcmp0(propertyName, "CurrentPageNumber"))
            return g_variant_new_int32(-1);
        if (!g_strcmp0(propertyName, "PageCount"))
            return g_variant_new_int32(-1);

        g_set_error(error, G_IO_ERROR, G_IO_ERROR_NOT_SUPPORTED, "Unknown property '%s'", propertyName);
        return nullptr;
    },
    // set_property,
    nullptr,
    // padding
    { nullptr }
};

String AccessibilityObjectAtspi::documentAttribute(const String& name) const
{
    if (!m_coreObject)
        return { };

    auto* document = m_coreObject->document();
    if (!document)
        return { };

    if (name == "DocType"_s)
        return document->doctype() ? document->doctype()->name() : String();
    if (name == "Encoding"_s)
        return document->charset();
    if (name == "URI"_s)
        return document->documentURI();
    if (name == "MimeType"_s)
        return document->contentType();
    if (name == "Title"_s)
        return document->title();

    return { };
}

UncheckedKeyHashMap<String, String> AccessibilityObjectAtspi::documentAttributes() const
{
    UncheckedKeyHashMap<String, String> map;
    if (!m_coreObject)
        return map;

    auto* document = m_coreObject->document();
    if (!document)
        return map;

    if (auto* doctype = document->doctype())
        map.add("DocType"_s, doctype->name());

    auto charset = document->charset();
    if (!charset.isEmpty())
        map.add("Encoding"_s, WTFMove(charset));

    auto uri = document->documentURI();
    if (!uri.isEmpty())
        map.add("URI"_s, WTFMove(uri));

    auto contentType = document->contentType();
    if (!contentType.isEmpty())
        map.add("MimeType"_s, WTFMove(contentType));

    const auto& title = document->title();
    if (!title.isEmpty())
        map.add("Title"_s, title);

    return map;
}

String AccessibilityObjectAtspi::documentLocale() const
{
    if (!m_coreObject)
        return { };

    auto* document = m_coreObject->document();
    if (!document)
        return { };

    return document->contentLanguage();
}

} // namespace WebCore

#endif // USE(ATSPI)
