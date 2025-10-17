/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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

#include "AccessibilityRootAtspi.h"
#include <gio/gio.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

GDBusInterfaceVTable AccessibilityObjectAtspi::s_actionFunctions = {
    // method_call
    [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar* methodName, GVariant* parameters, GDBusMethodInvocation* invocation, gpointer userData) {
        auto atspiObject = Ref { *static_cast<AccessibilityObjectAtspi*>(userData) };
        atspiObject->updateBackingStore();

        if (!g_strcmp0(methodName, "GetDescription"))
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(s)", ""));
        else if (!g_strcmp0(methodName, "GetName")) {
            int index;
            g_variant_get(parameters, "(i)", &index);
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(s)", !index ? atspiObject->actionName().utf8().data() : ""));
        } else if (!g_strcmp0(methodName, "GetLocalizedName")) {
            int index;
            g_variant_get(parameters, "(i)", &index);
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(s)", !index ? atspiObject->localizedActionName().utf8().data() : ""));
        } else if (!g_strcmp0(methodName, "GetKeyBinding")) {
            int index;
            g_variant_get(parameters, "(i)", &index);
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(s)", !index ? atspiObject->actionKeyBinding().utf8().data() : ""));
        } else if (!g_strcmp0(methodName, "DoAction")) {
            int index;
            g_variant_get(parameters, "(i)", &index);
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(b)", !index ? atspiObject->doAction() : FALSE));
        }
    },
    // get_property
    [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar* propertyName, GError** error, gpointer userData) -> GVariant* {
        auto atspiObject = Ref { *static_cast<AccessibilityObjectAtspi*>(userData) };
        atspiObject->updateBackingStore();

        if (!g_strcmp0(propertyName, "NActions"))
            return g_variant_new_int32(1);

        g_set_error(error, G_IO_ERROR, G_IO_ERROR_NOT_SUPPORTED, "Unknown property '%s'", propertyName);
        return nullptr;
    },
    // set_property,
    nullptr,
    // padding
    { nullptr }
};

String AccessibilityObjectAtspi::actionName() const
{
    return m_coreObject ? m_coreObject->actionVerb() : String();
}

String AccessibilityObjectAtspi::localizedActionName() const
{
    return m_coreObject ? m_coreObject->localizedActionVerb() : String();
}

String AccessibilityObjectAtspi::actionKeyBinding() const
{
    return m_coreObject ? m_coreObject->accessKey() : String();
}

bool AccessibilityObjectAtspi::doAction() const
{
    return m_coreObject ? m_coreObject->performDefaultAction() : false;
}

} // namespace WebCore

#endif // USE(ATSPI)
