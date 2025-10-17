/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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
#pragma once

#if ENABLE(WK_WEB_EXTENSIONS)

#include "WebExtensionAlarmParameters.h"
#include <WebCore/Timer.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/RefCounted.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WebExtensionContext;

class WebExtensionAlarm : public RefCounted<WebExtensionAlarm> {
    WTF_MAKE_NONCOPYABLE(WebExtensionAlarm);
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionAlarm);

public:
    template<typename... Args>
    static Ref<WebExtensionAlarm> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionAlarm(std::forward<Args>(args)...));
    }

    explicit WebExtensionAlarm(String name, Seconds initialInterval, Seconds repeatInterval, Function<void(WebExtensionAlarm&)>&& handler = nullptr)
        : m_parameters({ name, initialInterval, repeatInterval, MonotonicTime::nan() })
        , m_handler(WTFMove(handler))
    {
        ASSERT(!name.isNull());
        schedule();
    }

    const WebExtensionAlarmParameters& parameters() const { return m_parameters; }

    const String& name() const { return m_parameters.name; }
    Seconds initialInterval() const { return m_parameters.initialInterval; }
    Seconds repeatInterval() const { return m_parameters.repeatInterval; }
    MonotonicTime nextScheduledTime() const { return m_parameters.nextScheduledTime; }

private:
    void schedule();
    void fire();

    WebExtensionAlarmParameters m_parameters;

    Function<void(WebExtensionAlarm&)> m_handler;
    std::unique_ptr<RunLoop::Timer> m_timer;
    bool m_hasFiredInitialTimer { false };
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
