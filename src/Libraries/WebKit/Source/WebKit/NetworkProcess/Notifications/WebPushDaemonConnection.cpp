/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
#include "WebPushDaemonConnection.h"

#if ENABLE(WEB_PUSH_NOTIFICATIONS)

#include "DaemonDecoder.h"
#include "DaemonEncoder.h"
#include "NetworkSession.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebPushD {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Connection);

Ref<Connection> Connection::create(CString&& machServiceName, WebPushDaemonConnectionConfiguration&& configuration)
{
    return adoptRef(*new Connection(WTFMove(machServiceName), WTFMove(configuration)));
}

Connection::Connection(CString&& machServiceName, WebPushDaemonConnectionConfiguration&& configuration)
    : Daemon::ConnectionToMachService<ConnectionTraits>(WTFMove(machServiceName))
    , m_configuration(WTFMove(configuration))
{
    LOG(Push, "Creating WebPushD connection to mach service: %s", this->machServiceName().data());
}

} // namespace WebKit::WebPushD

#endif // ENABLE(WEB_PUSH_NOTIFICATIONS)
