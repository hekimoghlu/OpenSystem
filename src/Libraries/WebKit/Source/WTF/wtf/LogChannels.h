/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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

#include <optional>
#include <wtf/Assertions.h>
#include <wtf/Forward.h>
#include <wtf/text/WTFString.h>

namespace WTF {

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED

class LogChannels {
public:
    virtual ~LogChannels() = default;
    virtual String logLevelString() = 0;

    WTF_EXPORT_PRIVATE bool isLogChannelEnabled(const String& name);
    WTF_EXPORT_PRIVATE void setLogChannelToAccumulate(const String& name);
    WTF_EXPORT_PRIVATE void clearAllLogChannelsToAccumulate();
    WTF_EXPORT_PRIVATE void initializeLogChannelsIfNecessary(std::optional<String> = std::nullopt);
    WTF_EXPORT_PRIVATE WTFLogChannel* getLogChannel(const String& name);

protected:
    Vector<WTFLogChannel*> m_logChannels;
    bool m_logChannelsNeedInitialization { true };
};

#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED

} // namespace WTF
