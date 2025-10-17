/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#include <wtf/glib/ChassisType.h>

#include <mutex>
#include <optional>
#include <wtf/glib/GUniquePtr.h>

namespace WTF {

static std::optional<ChassisType> readMachineInfoChassisType()
{
    GUniqueOutPtr<char> buffer;
    GUniqueOutPtr<GError> error;
    if (!g_file_get_contents("/etc/machine-info", &buffer.outPtr(), nullptr, &error.outPtr())) {
        if (!g_error_matches(error.get(), G_FILE_ERROR, G_FILE_ERROR_NOENT))
            g_warning("Could not open /etc/machine-info: %s", error->message);
        return std::nullopt;
    }

    GUniquePtr<char*> split(g_strsplit(buffer.get(), "\n", -1));
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GLib port.
    for (int i = 0; split.get()[i]; ++i) {
        if (g_str_has_prefix(split.get()[i], "CHASSIS=")) {
            char* chassis = split.get()[i] + 8;

            GUniquePtr<char> unquoted(g_shell_unquote(chassis, &error.outPtr()));
            if (error)
                g_warning("Could not unquote chassis type %s: %s", chassis, error->message);

            if (!strcmp(unquoted.get(), "tablet") || !strcmp(unquoted.get(), "handset"))
                return ChassisType::Mobile;

            return ChassisType::Desktop;
        }
    }
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    return std::nullopt;
}

static std::optional<ChassisType> readDMIChassisType()
{
    GUniqueOutPtr<char> buffer;
    GUniqueOutPtr<GError> error;
    if (g_file_get_contents("/sys/class/dmi/id/chassis_type", &buffer.outPtr(), nullptr, &error.outPtr())) {
        int type = strtol(buffer.get(), nullptr, 10);

        // See the SMBIOS Specification 3.0 section 7.4.1 for details about the values listed here:
        // https://www.dmtf.org/sites/default/files/standards/documents/DSP0134_3.0.0.pdf
        switch (type) {
        case 0x3: /* Desktop */
        case 0x4: /* Low Profile Desktop */
        case 0x6: /* Mini Tower */
        case 0x7: /* Tower */
        case 0x8: /* Portable */
        case 0x9: /* Laptop */
        case 0xA: /* Notebook */
        case 0xE: /* Sub Notebook */
        case 0x11: /* Main Server Chassis */
        case 0x1C: /* Blade */
        case 0x1D: /* Blade Enclosure */
        case 0x1F: /* Convertible */
        case 0x20: /* Detachable */
            return ChassisType::Desktop;

        case 0xB: /* Hand Held */
        case 0x1E: /* Tablet */
            return ChassisType::Mobile;
        }
    } else if (!g_error_matches(error.get(), G_FILE_ERROR, G_FILE_ERROR_NOENT))
        g_warning("Could not open /sys/class/dmi/id/chassis_type: %s", error->message);

    return std::nullopt;
}

static std::optional<ChassisType> readACPIChassisType()
{
    GUniqueOutPtr<char> buffer;
    GUniqueOutPtr<GError> error;
    if (g_file_get_contents("/sys/firmware/acpi/pm_profile", &buffer.outPtr(), nullptr, &error.outPtr())) {
        int type = strtol(buffer.get(), nullptr, 10);

        // See the ACPI 5.0 Spec Section 5.2.9.1 for details:
        // http://www.acpi.info/DOWNLOADS/ACPIspec50.pdf
        switch (type) {
        case 1: /* Desktop */
        case 2: /* Mobile */
        case 3: /* Workstation */
        case 4: /* Enterprise Server */
        case 5: /* SOHO Server */
        case 6: /* Appliance PC */
        case 7: /* Performance Server */
            return ChassisType::Desktop;

        case 8: /* Tablet */
            return ChassisType::Mobile;
        }
    } else if (!g_error_matches(error.get(), G_FILE_ERROR, G_FILE_ERROR_NOENT))
        g_warning("Could not open /sys/firmware/acpi/pm_profile: %s", error->message);

    return std::nullopt;
}

ChassisType chassisType()
{
    static ChassisType chassisType;
    static std::once_flag initializeChassis;
    std::call_once(initializeChassis, [] {
        auto optionalChassisType = readMachineInfoChassisType();
        if (!optionalChassisType)
            optionalChassisType = readDMIChassisType();
        if (!optionalChassisType)
            optionalChassisType = readACPIChassisType();
        chassisType = optionalChassisType.value_or(ChassisType::Desktop);
    });

    return chassisType;
}

} // namespace WTF
