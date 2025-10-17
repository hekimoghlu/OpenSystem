/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
#include "DisplayVBlankMonitorTimer.h"

#include <WebCore/AnimationFrameRate.h>
#include <chrono>
#include <thread>

namespace WebKit {

std::unique_ptr<DisplayVBlankMonitor> DisplayVBlankMonitorTimer::create()
{
    return makeUnique<DisplayVBlankMonitorTimer>();
}

DisplayVBlankMonitorTimer::DisplayVBlankMonitorTimer()
    : DisplayVBlankMonitor(WebCore::FullSpeedFramesPerSecond)
{
}

bool DisplayVBlankMonitorTimer::waitForVBlank() const
{
    std::this_thread::sleep_for(std::chrono::milliseconds(1000 / m_refreshRate));
    return true;
}

} // namespace WebKit
