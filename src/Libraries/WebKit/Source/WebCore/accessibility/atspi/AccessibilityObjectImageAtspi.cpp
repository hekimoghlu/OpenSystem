/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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

#include "AccessibilityAtspiEnums.h"
#include "AccessibilityObject.h"
#include <gio/gio.h>

namespace WebCore {

GDBusInterfaceVTable AccessibilityObjectAtspi::s_imageFunctions = {
    // method_call
    [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar* methodName, GVariant* parameters, GDBusMethodInvocation* invocation, gpointer userData) {
        auto atspiObject = Ref { *static_cast<AccessibilityObjectAtspi*>(userData) };
        atspiObject->updateBackingStore();

        if (!g_strcmp0(methodName, "GetImageExtents")) {
            uint32_t coordinateType;
            g_variant_get(parameters, "(u)", &coordinateType);
            auto rect = atspiObject->elementRect(static_cast<Atspi::CoordinateType>(coordinateType));
            g_dbus_method_invocation_return_value(invocation, g_variant_new("((iiii))", rect.x(), rect.y(), rect.width(), rect.height()));
        } else if (!g_strcmp0(methodName, "GetImagePosition")) {
            uint32_t coordinateType;
            g_variant_get(parameters, "(u)", &coordinateType);
            auto rect = atspiObject->elementRect(static_cast<Atspi::CoordinateType>(coordinateType));
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(ii)", rect.x(), rect.y()));
        } else if (!g_strcmp0(methodName, "GetImageSize")) {
            auto rect = atspiObject->elementRect(Atspi::CoordinateType::ParentCoordinates);
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(ii)", rect.width(), rect.height()));
        }
    },
    // get_property
    [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar* propertyName, GError** error, gpointer userData) -> GVariant* {
        auto atspiObject = Ref { *static_cast<AccessibilityObjectAtspi*>(userData) };
        atspiObject->updateBackingStore();

        if (!g_strcmp0(propertyName, "ImageDescription"))
            return g_variant_new_string(atspiObject->imageDescription().utf8().data());
        if (!g_strcmp0(propertyName, "ImageLocale"))
            return g_variant_new_string(atspiObject->locale().utf8().data());

        g_set_error(error, G_IO_ERROR, G_IO_ERROR_NOT_SUPPORTED, "Unknown property '%s'", propertyName);
        return nullptr;
    },
    // set_property,
    nullptr,
    // padding
    { nullptr }
};

String AccessibilityObjectAtspi::imageDescription() const
{
    if (!m_coreObject)
        return { };

    Vector<AccessibilityText> textOrder;
    m_coreObject->accessibilityText(textOrder);

    bool visibleTextAvailable = false;
    for (const AccessibilityText& text : textOrder) {
        if (text.textSource == AccessibilityTextSource::Alternative)
            return text.text;

        switch (text.textSource) {
        case AccessibilityTextSource::Visible:
        case AccessibilityTextSource::Children:
        case AccessibilityTextSource::LabelByElement:
            visibleTextAvailable = true;
        default:
            break;
        }

        if (text.textSource == AccessibilityTextSource::TitleTag && !visibleTextAvailable)
            return text.text;
    }

    return { };
}

} // namespace WebCore

#endif // USE(ATSPI)
