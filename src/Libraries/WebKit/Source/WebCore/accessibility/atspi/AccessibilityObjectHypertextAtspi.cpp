/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

GDBusInterfaceVTable AccessibilityObjectAtspi::s_hypertextFunctions = {
    // method_call
    [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar* methodName, GVariant* parameters, GDBusMethodInvocation* invocation, gpointer userData) {
        auto atspiObject = Ref { *static_cast<AccessibilityObjectAtspi*>(userData) };
        atspiObject->updateBackingStore();

        if (!g_strcmp0(methodName, "GetNLinks")) {
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(i)", atspiObject->hyperlinkCount()));
        } else if (!g_strcmp0(methodName, "GetLink")) {
            int index;
            g_variant_get(parameters, "(i)", &index);
            auto* wrapper = index >= 0 ? atspiObject->hyperlink(index) : nullptr;
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(@(so))", wrapper ? wrapper->hyperlinkReference() : AccessibilityAtspi::singleton().nullReference()));
        } else if (!g_strcmp0(methodName, "GetLinkIndex")) {
            int offset;
            g_variant_get(parameters, "(i)", &offset);
            g_dbus_method_invocation_return_value(invocation, g_variant_new("(i)", offset >= 0 ? atspiObject->hyperlinkIndex(offset).value_or(-1) : -1));
        }
    },
    // get_property
    nullptr,
    // set_property,
    nullptr,
    // padding
    { nullptr }
};

unsigned AccessibilityObjectAtspi::hyperlinkCount() const
{
    if (!m_coreObject)
        return 0;

    unsigned linkCount = 0;
    const auto& children = m_coreObject->children();
    for (const auto& child : children) {
        if (child->isIgnored())
            continue;

        auto* wrapper = child->wrapper();
        if (wrapper && wrapper->interfaces().contains(Interface::Hyperlink))
            linkCount++;
    }

    return linkCount;
}

AccessibilityObjectAtspi* AccessibilityObjectAtspi::hyperlink(unsigned index) const
{
    if (!m_coreObject)
        return nullptr;

    const auto& children = m_coreObject->children();
    if (index >= children.size())
        return nullptr;

    int linkIndex = -1;
    for (const auto& child : children) {
        if (child->isIgnored())
            continue;

        auto* wrapper = child->wrapper();
        if (!wrapper || !wrapper->interfaces().contains(Interface::Hyperlink))
            continue;

        if (static_cast<unsigned>(++linkIndex) == index)
            return wrapper;
    }

    return nullptr;
}

std::optional<unsigned> AccessibilityObjectAtspi::hyperlinkIndex(unsigned offset) const
{
    return characterIndex(objectReplacementCharacter, offset);
}

} // namespace WebCore

#endif // USE(ATSPI)
