/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
#include "LowPowerModeNotifier.h"

#include <cstring>
#include <gio/gio.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebCore {


LowPowerModeNotifier::LowPowerModeNotifier(LowPowerModeChangeCallback&& callback)
    : m_callback(WTFMove(callback))
    , m_powerProfileMonitor(adoptGRef(g_power_profile_monitor_dup_default()))
    , m_lowPowerModeEnabled(g_power_profile_monitor_get_power_saver_enabled(m_powerProfileMonitor.get()))
{
    g_signal_connect_swapped(m_powerProfileMonitor.get(), "notify::power-saver-enabled", G_CALLBACK(+[] (LowPowerModeNotifier* self, GParamSpec*, GPowerProfileMonitor* monitor) {
        bool powerSaverEnabled = g_power_profile_monitor_get_power_saver_enabled(monitor);
        if (self->m_lowPowerModeEnabled != powerSaverEnabled) {
            self->m_lowPowerModeEnabled = powerSaverEnabled;
            self->m_callback(self->m_lowPowerModeEnabled);
        }
    }), this);
}

LowPowerModeNotifier::~LowPowerModeNotifier()
{
    g_signal_handlers_disconnect_by_data(m_powerProfileMonitor.get(), this);
}

bool LowPowerModeNotifier::isLowPowerModeEnabled() const
{
    return m_lowPowerModeEnabled;
}

} // namespace WebCore
