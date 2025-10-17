/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
#include "AccessibilityObject.h"
#include <gio/gio.h>

namespace WebCore {

GDBusInterfaceVTable AccessibilityObjectAtspi::s_valueFunctions = {
    // method_call
    nullptr,
    // get_property
    [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar* propertyName, GError** error, gpointer userData) -> GVariant* {
        auto atspiObject = Ref { *static_cast<AccessibilityObjectAtspi*>(userData) };
        atspiObject->updateBackingStore();

        if (!g_strcmp0(propertyName, "CurrentValue"))
            return g_variant_new_double(atspiObject->currentValue());
        if (!g_strcmp0(propertyName, "MinimumValue"))
            return g_variant_new_double(atspiObject->minimumValue());
        if (!g_strcmp0(propertyName, "MaximumValue"))
            return g_variant_new_double(atspiObject->maximumValue());
        if (!g_strcmp0(propertyName, "MinimumIncrement"))
            return g_variant_new_double(atspiObject->minimumIncrement());

        g_set_error(error, G_IO_ERROR, G_IO_ERROR_NOT_SUPPORTED, "Unknown property '%s'", propertyName);
        return nullptr;
    },
    // set_property,
    [](GDBusConnection*, const gchar*, const gchar*, const gchar*, const gchar* propertyName, GVariant* propertyValue, GError** error, gpointer userData) -> gboolean {
        auto atspiObject = Ref { *static_cast<AccessibilityObjectAtspi*>(userData) };
        atspiObject->updateBackingStore();

        if (!g_strcmp0(propertyName, "CurrentValue"))
            return atspiObject->setCurrentValue(g_variant_get_double(propertyValue));

        g_set_error(error, G_IO_ERROR, G_IO_ERROR_NOT_SUPPORTED, "Unknown property '%s'", propertyName);
        return FALSE;
    },
    // padding
    { nullptr }
};

double AccessibilityObjectAtspi::currentValue() const
{
    return m_coreObject ? m_coreObject->valueForRange() : 0;
}

bool AccessibilityObjectAtspi::setCurrentValue(double value)
{
    if (!m_coreObject)
        return false;

    if (!m_coreObject->canSetValueAttribute())
        return false;

    if (m_coreObject->canSetNumericValue())
        return m_coreObject->setValue(value);

    return m_coreObject->setValue(String::numberToStringFixedPrecision(value));
}

double AccessibilityObjectAtspi::minimumValue() const
{
    return m_coreObject ? m_coreObject->minValueForRange() : 0;
}

double AccessibilityObjectAtspi::maximumValue() const
{
    return m_coreObject ? m_coreObject->maxValueForRange() : 0;
}

double AccessibilityObjectAtspi::minimumIncrement() const
{
    if (!m_coreObject)
        return 0;

    auto stepAttribute = static_cast<AccessibilityObject*>(m_coreObject.get())->getAttribute(HTMLNames::stepAttr);
    if (!stepAttribute.isEmpty())
        return stepAttribute.toFloat();

    // If 'step' attribute is not defined, WebCore assumes a 5% of the range between
    // minimum and maximum values. Implicit value of step should be one or larger.
    float step = (m_coreObject->maxValueForRange() - m_coreObject->minValueForRange()) * 0.05;
    return step < 1 ? 1 : step;
}

void AccessibilityObjectAtspi::valueChanged(double value)
{
    AccessibilityAtspi::singleton().valueChanged(*this, value);
}

} // namespace WebCore

#endif // USE(ATSPI)
