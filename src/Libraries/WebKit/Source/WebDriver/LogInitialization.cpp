/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#include "LogInitialization.h"

#include "Logging.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/text/WTFString.h>

namespace WebDriver {

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED

class LogChannels final : public WTF::LogChannels {
public:
    LogChannels()
    {
        m_logChannels = {
            WEBDRIVER_LOG_CHANNELS(LOG_CHANNEL_ADDRESS)
        };
    }

private:
    String logLevelString() final
    {
        return WebDriver::logLevelString();
    }
};

WTF::LogChannels& logChannels()
{
    static NeverDestroyed<LogChannels> logChannels;
    return logChannels.get();
}

WTFLogChannel* getLogChannel(const String& name)
{
    return logChannels().getLogChannel(name);
}

#else

WTFLogChannel* getLogChannel(const String&)
{
    return nullptr;
}

#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED

} // namespace WebDriver
