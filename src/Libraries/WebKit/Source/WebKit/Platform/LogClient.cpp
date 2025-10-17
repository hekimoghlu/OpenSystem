/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
#include "LogClient.h"

#if ENABLE(LOGD_BLOCKING_IN_WEBCONTENT)

namespace WebKit {

LogClient::LogClient(IPC::StreamClientConnection& connection, const LogStreamIdentifier &identifier)
    : m_logStreamConnection(connection)
    , m_logStreamIdentifier(identifier)
{
}

void LogClient::log(std::span<const uint8_t> logChannel, std::span<const uint8_t> logCategory, std::span<const uint8_t> logString, os_log_type_t type)
{
    Locker locker { m_logStreamLock };
    m_logStreamConnection->send(Messages::LogStream::LogOnBehalfOfWebContent(logChannel, logCategory, logString, type), m_logStreamIdentifier);
}

}

#endif // ENABLE(LOGD_BLOCKING_IN_WEBCONTENT)
