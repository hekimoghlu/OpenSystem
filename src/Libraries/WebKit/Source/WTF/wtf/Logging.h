/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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

#include <wtf/Assertions.h>
#include <wtf/Forward.h>

namespace WTF {

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED

#ifndef LOG_CHANNEL_PREFIX
#define LOG_CHANNEL_PREFIX WTFLog
#endif

#define WTF_LOG_CHANNELS(M) \
    M(Language) \
    M(RefCountedLeaks) \
    M(Process) \
    M(Threading) \
    M(MemoryPressure) \
    M(SuspendableWorkQueue) \
    M(NativePromise) \

#undef DECLARE_LOG_CHANNEL
#define DECLARE_LOG_CHANNEL(name) \
    WTF_EXPORT_PRIVATE extern WTFLogChannel JOIN_LOG_CHANNEL_WITH_PREFIX(LOG_CHANNEL_PREFIX, name);

WTF_LOG_CHANNELS(DECLARE_LOG_CHANNEL)

#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED

} // namespace WTF
