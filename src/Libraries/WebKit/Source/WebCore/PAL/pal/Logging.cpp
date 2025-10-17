/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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

#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
#include <notify.h>
#include <wtf/BlockPtr.h>
#endif

namespace PAL {

void registerNotifyCallback(ASCIILiteral notifyID, Function<void()>&& callback)
{
#if PLATFORM(COCOA)
    int token;
    notify_register_dispatch(notifyID.characters(), &token, dispatch_get_main_queue(), makeBlockPtr([callback = WTFMove(callback)](int) {
        callback();
    }).get());
#else
    UNUSED_PARAM(notifyID);
    UNUSED_PARAM(callback);
#endif
}

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED

#define DEFINE_PAL_LOG_CHANNEL(name) DEFINE_LOG_CHANNEL(name, LOG_CHANNEL_WEBKIT_SUBSYSTEM)
PAL_LOG_CHANNELS(DEFINE_PAL_LOG_CHANNEL)

#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED

} // namespace WebCore
