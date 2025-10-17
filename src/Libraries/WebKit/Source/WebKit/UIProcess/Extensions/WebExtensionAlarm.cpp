/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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
#include "WebExtensionAlarm.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(WK_WEB_EXTENSIONS)

#include "Logging.h"

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebExtensionAlarm);

void WebExtensionAlarm::schedule()
{
    m_parameters.nextScheduledTime = MonotonicTime::now() + initialInterval();

    RELEASE_LOG_DEBUG(Extensions, "Scheduled alarm; initial = %{public}f seconds; repeat = %{public}f seconds", initialInterval().seconds(), repeatInterval().seconds());

    m_timer = makeUnique<RunLoop::Timer>(RunLoop::current(), this, &WebExtensionAlarm::fire);
    m_timer->startOneShot(initialInterval());
}

void WebExtensionAlarm::fire()
{
    // Calculate the next scheduled time now, so the handler's work time does not count against it.
    auto nextScheduledTime = MonotonicTime::now() + repeatInterval();
    if (!m_hasFiredInitialTimer) {
        m_hasFiredInitialTimer = true;
        m_timer->startRepeating(repeatInterval());
    }

    m_handler(*this);

    if (!repeatInterval()) {
        m_timer = nullptr;
        return;
    }

    m_parameters.nextScheduledTime = nextScheduledTime;
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
