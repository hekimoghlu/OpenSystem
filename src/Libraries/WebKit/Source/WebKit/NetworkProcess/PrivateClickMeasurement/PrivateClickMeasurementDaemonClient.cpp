/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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
#include "PrivateClickMeasurementDaemonClient.h"
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include "PCMDaemonConnectionSet.h"
#endif

namespace WebKit::PCM {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DaemonClient);

void DaemonClient::broadcastConsoleMessage(JSC::MessageLevel level, const String& message)
{
#if PLATFORM(COCOA)
    DaemonConnectionSet::singleton().broadcastConsoleMessage(level, message);
#else
    UNUSED_PARAM(level);
    UNUSED_PARAM(message);
#endif
}

bool DaemonClient::featureEnabled() const
{
    return true;
}

bool DaemonClient::debugModeEnabled() const
{
#if PLATFORM(COCOA)
    return DaemonConnectionSet::singleton().debugModeEnabled();
#else
    return false;
#endif
}

} // namespace WebKit
