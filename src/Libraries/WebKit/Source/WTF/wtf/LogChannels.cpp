/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#include "LogChannels.h"

#include <wtf/LoggingAccumulator.h>

namespace WTF {

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED

bool LogChannels::isLogChannelEnabled(const String& name)
{
    WTFLogChannel* channel = getLogChannel(name);
    if (!channel)
        return false;
    return channel->state != WTFLogChannelState::Off;
}

void LogChannels::setLogChannelToAccumulate(const String& name)
{
    WTFLogChannel* channel = getLogChannel(name);
    if (!channel)
        return;

    channel->state = WTFLogChannelState::OnWithAccumulation;
    m_logChannelsNeedInitialization = true;
}

void LogChannels::clearAllLogChannelsToAccumulate()
{
    resetAccumulatedLogs();
    for (auto* channel : m_logChannels) {
        if (channel->state == WTFLogChannelState::OnWithAccumulation)
            channel->state = WTFLogChannelState::Off;
    }

    m_logChannelsNeedInitialization = true;
}

void LogChannels::initializeLogChannelsIfNecessary(std::optional<String> logChannelString)
{
    if (!m_logChannelsNeedInitialization && !logChannelString)
        return;

    m_logChannelsNeedInitialization = false;

    String enabledChannelsString = logChannelString ? logChannelString.value() : logLevelString();
    WTFInitializeLogChannelStatesFromString(m_logChannels.data(), m_logChannels.size(), enabledChannelsString.utf8().data());
}

WTFLogChannel* LogChannels::getLogChannel(const String& name)
{
    return WTFLogChannelByName(m_logChannels.data(), m_logChannels.size(), name.utf8().data());
}

#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED

} // namespace WTF
