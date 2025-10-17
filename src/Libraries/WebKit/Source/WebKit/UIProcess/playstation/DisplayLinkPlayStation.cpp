/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#include "DisplayLink.h"

#if HAVE(DISPLAY_LINK)

#include <wpe/playstation.h>

namespace WebKit {

void DisplayLink::platformInitialize()
{
    static const struct wpe_playstation_display_client_interface s_client = {
        // vblank
        [](void* data) {
            static_cast<DisplayLink*>(data)->notifyObserversDisplayDidRefresh();
        },
        nullptr, nullptr, nullptr,
    };

    m_display = wpe_playstation_display_create();
    wpe_playstation_display_set_client(m_display, &s_client, this);
}

void DisplayLink::platformFinalize()
{
    wpe_playstation_display_destroy(m_display);
}

bool DisplayLink::platformIsRunning() const
{
    return wpe_playstation_display_is_running(m_display);
}

void DisplayLink::platformStart()
{
    wpe_playstation_display_start(m_display);
}

void DisplayLink::platformStop()
{
    wpe_playstation_display_stop(m_display);
}

} // namespace WebKit

#endif
