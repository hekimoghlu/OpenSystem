/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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

#include "AccessibilityAtspi.h"
#include <gio/gio.h>
#include <wtf/URL.h>
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

GDBusInterfaceVTable AccessibilityObjectAtspi::s_hyperlinkFunctions = {
    // method_call
    [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar* methodName, GVariant* parameters, GDBusMethodInvocation* invocation, gpointer userData) {
        auto atspiObject = Ref { *static_cast<AccessibilityObjectAtspi*>(userData) };
        atspiObject->updateBackingStore();

        if (!g_strcmp0(methodName, "GetObject")) {
            int index;
            g_variant_get(parameters, "(i)", &index);
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(@(so))", !index ? atspiObject->reference() : AccessibilityAtspi::singleton().nullReference()));
        } else if (!g_strcmp0(methodName, "GetURI")) {
            int index;
            g_variant_get(parameters, "(i)", &index);
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(s)", !index ? atspiObject->url().string().utf8().data() : ""));
        } else if (!g_strcmp0(methodName, "IsValid"))
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(b)", atspiObject->m_coreObject ? TRUE : FALSE));
    },
    // get_property
    [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar* propertyName, GError** error, gpointer userData) -> GVariant* {
        auto atspiObject = Ref { *static_cast<AccessibilityObjectAtspi*>(userData) };
        atspiObject->updateBackingStore();

        if (!g_strcmp0(propertyName, "NAnchors"))
            return g_variant_new_int32(1);
        if (!g_strcmp0(propertyName, "StartIndex"))
            return g_variant_new_int32(atspiObject->offsetInParent());
        if (!g_strcmp0(propertyName, "EndIndex"))
            return g_variant_new_int32(atspiObject->offsetInParent() + 1);

        g_set_error(error, G_IO_ERROR, G_IO_ERROR_NOT_SUPPORTED, "Unknown property '%s'", propertyName);
        return nullptr;
    },
    // set_property,
    nullptr,
    // padding
    { nullptr }
};

URL AccessibilityObjectAtspi::url() const
{
    return m_coreObject ? m_coreObject->url() : URL();
}

unsigned AccessibilityObjectAtspi::offsetInParent() const
{
    if (!m_coreObject)
        return 0;

    auto* parent = m_coreObject->parentObjectUnignored();
    if (!parent || !parent->wrapper())
        return 0;

    int index = -1;
    const auto& children = parent->children();
    for (const auto& child : children) {
        if (child->isIgnored())
            continue;

        auto* wrapper = child->wrapper();
        if (!wrapper || !wrapper->interfaces().contains(Interface::Hyperlink))
            continue;

        index++;
        if (wrapper == this)
            break;
    }

    if (index == -1)
        return 0;

    return parent->wrapper()->characterOffset(objectReplacementCharacter, index).value_or(0);
}

} // namespace WebCore

#endif // USE(ATSPI)
