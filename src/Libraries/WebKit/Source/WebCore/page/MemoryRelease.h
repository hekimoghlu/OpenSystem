/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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

#include <wtf/MemoryPressureHandler.h>

namespace WebCore {

enum class MaintainBackForwardCache : bool { No, Yes };
enum class MaintainMemoryCache : bool { No, Yes };
enum class LogMemoryStatisticsReason : uint8_t {
    DebugNotification,
    WarningMemoryPressureNotification,
    CriticalMemoryPressureNotification,
    OutOfMemoryDeath
};

WEBCORE_EXPORT void releaseMemory(Critical, Synchronous, MaintainBackForwardCache = MaintainBackForwardCache::No, MaintainMemoryCache = MaintainMemoryCache::No);
void platformReleaseMemory(Critical);
WEBCORE_EXPORT void releaseGraphicsMemory(Critical, Synchronous);
void platformReleaseGraphicsMemory(Critical);
void jettisonExpensiveObjectsOnTopLevelNavigation();
WEBCORE_EXPORT void registerMemoryReleaseNotifyCallbacks();
WEBCORE_EXPORT void logMemoryStatistics(LogMemoryStatisticsReason);

} // namespace WebCore
