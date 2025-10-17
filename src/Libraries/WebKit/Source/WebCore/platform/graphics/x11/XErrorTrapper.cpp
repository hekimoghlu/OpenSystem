/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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
#include "XErrorTrapper.h"

#if PLATFORM(X11)
#include <sys/types.h>
#include <unistd.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

static UncheckedKeyHashMap<Display*, Vector<XErrorTrapper*>>& xErrorTrappersMap()
{
    static NeverDestroyed<UncheckedKeyHashMap<Display*, Vector<XErrorTrapper*>>> trappersMap;
    return trappersMap;
}

XErrorTrapper::XErrorTrapper(Display* display, Policy policy, Vector<unsigned char>&& expectedErrors)
    : m_display(display)
    , m_policy(policy)
    , m_expectedErrors(WTFMove(expectedErrors))
{
    xErrorTrappersMap().add(m_display, Vector<XErrorTrapper*>()).iterator->value.append(this);
    m_previousErrorHandler = XSetErrorHandler([](Display* display, XErrorEvent* event) -> int {
        auto iterator = xErrorTrappersMap().find(display);
        if (iterator == xErrorTrappersMap().end())
            return 0;

        ASSERT(!iterator->value.isEmpty());
        iterator->value.last()->errorEvent(event);
        return 0;
    });
}

XErrorTrapper::~XErrorTrapper()
{
    XSync(m_display, False);
    auto iterator = xErrorTrappersMap().find(m_display);
    ASSERT(iterator != xErrorTrappersMap().end());
    auto* trapper = iterator->value.takeLast();
    ASSERT_UNUSED(trapper, trapper == this);
    if (iterator->value.isEmpty())
        xErrorTrappersMap().remove(iterator);

    XSetErrorHandler(m_previousErrorHandler);
}

unsigned char XErrorTrapper::errorCode() const
{
    XSync(m_display, False);
    return m_errorCode;
}

void XErrorTrapper::errorEvent(XErrorEvent* event)
{
    m_errorCode = event->error_code;
    if (m_policy == Policy::Ignore)
        return;

    if (m_expectedErrors.contains(m_errorCode))
        return;

    static const char errorFormatString[] = "The program with pid %d received an X Window System error.\n"
        "The error was '%s'.\n"
        "  (Details: serial %ld error_code %d request_code %d minor_code %d)\n";
    char errorMessage[64];
    XGetErrorText(m_display, m_errorCode, errorMessage, 63);
    WTFLogAlways(errorFormatString, getpid(), errorMessage, event->serial, event->error_code, event->request_code, event->minor_code);

    if (m_policy == Policy::Crash)
        CRASH();
}

} // namespace WebCore

#endif // PLATFORM(X11)
