/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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
#include "WebKitLogInitialization.h"

#include "WebKitLogging.h"
#include <wtf/text/CString.h>

namespace WebKit {

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED

class LogChannels final : public WTF::LogChannels {
public:
    LogChannels()
    {
        m_logChannels = {
            WEBKIT_LOG_CHANNELS(LOG_CHANNEL_ADDRESS)
        };
    }

private:
    String logLevelString() final
    {
        static NSString * const defaultsDomain = @"WebKitLogging";
        return [[NSUserDefaults standardUserDefaults] stringForKey:defaultsDomain];
    }
};

WTF::LogChannels& logChannels()
{
    static NeverDestroyed<LogChannels> logChannels;
    return logChannels.get();
}

#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED

} // namespace WebKit

void ReportDiscardedDelegateException(SEL delegateSelector, id exception)
{
    if ([exception isKindOfClass:[NSException class]]) {
        NSLog(@"*** WebKit discarded an uncaught exception in the %s delegate: <%@> %@",
            sel_getName(delegateSelector), [exception name], [exception reason]);
    } else {
        NSLog(@"*** WebKit discarded an uncaught exception in the %s delegate: %@",
            sel_getName(delegateSelector), exception);
    }
}
