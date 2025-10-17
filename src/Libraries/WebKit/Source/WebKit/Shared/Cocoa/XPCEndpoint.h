/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#include <wtf/OSObjectPtr.h>
#include <wtf/spi/darwin/XPCSPI.h>
#include <wtf/text/ASCIILiteral.h>

namespace WebKit {

class XPCEndpoint {
public:
    WK_EXPORT XPCEndpoint();
    virtual ~XPCEndpoint() = default;

    WK_EXPORT void sendEndpointToConnection(xpc_connection_t);

    WK_EXPORT OSObjectPtr<xpc_endpoint_t> endpoint() const;

    static constexpr auto xpcMessageNameKey = "message-name"_s;

private:
    virtual ASCIILiteral xpcEndpointMessageNameKey() const = 0;
    virtual ASCIILiteral xpcEndpointMessageName() const = 0;
    virtual ASCIILiteral xpcEndpointNameKey() const = 0;
    virtual void handleEvent(xpc_connection_t, xpc_object_t) = 0;

    OSObjectPtr<xpc_connection_t> m_connection;
    OSObjectPtr<xpc_endpoint_t> m_endpoint;
};

}

#endif
