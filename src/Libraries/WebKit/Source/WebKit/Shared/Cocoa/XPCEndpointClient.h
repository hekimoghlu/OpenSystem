/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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

#ifdef __cplusplus

#include <WebKit/WKBase.h>
#include <wtf/Lock.h>
#include <wtf/OSObjectPtr.h>
#include <wtf/spi/darwin/XPCSPI.h>

namespace WebKit {

class XPCEndpointClient {
public:
    virtual ~XPCEndpointClient() { }

    WK_EXPORT void setEndpoint(xpc_endpoint_t);

protected:
    WK_EXPORT OSObjectPtr<xpc_connection_t> connection();

private:
    virtual void handleEvent(xpc_object_t) = 0;
    virtual void didConnect() = 0;

    Lock m_connectionLock;
    OSObjectPtr<xpc_connection_t> m_connection WTF_GUARDED_BY_LOCK(m_connectionLock);
};

}

#endif
