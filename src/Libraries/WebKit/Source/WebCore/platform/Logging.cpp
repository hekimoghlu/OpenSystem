/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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
#include "Logging.h"

namespace WebCore {

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED

#if ENABLE(LOGD_BLOCKING_IN_WEBCONTENT)
std::unique_ptr<LogClient>& logClient()
{
    static NeverDestroyed<std::unique_ptr<LogClient>> client;
    return client;
}
#endif

#define DEFINE_WEBCORE_LOG_CHANNEL(name) DEFINE_LOG_CHANNEL(name, LOG_CHANNEL_WEBKIT_SUBSYSTEM)
WEBCORE_LOG_CHANNELS(DEFINE_WEBCORE_LOG_CHANNEL)

#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED

} // namespace WebCore
