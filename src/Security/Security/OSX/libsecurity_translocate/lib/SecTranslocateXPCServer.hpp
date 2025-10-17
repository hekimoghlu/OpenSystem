/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
/* Purpose: This defines the xpc server for translocation */

#ifndef SecTranslocateXPCServer_hpp
#define SecTranslocateXPCServer_hpp

#include <dispatch/dispatch.h>
#include <xpc/xpc.h>

namespace Security {
namespace SecTranslocate {

class XPCServer
{
public:
    XPCServer(dispatch_queue_t q);
    ~XPCServer();
private:
    XPCServer() = delete;
    XPCServer(const XPCServer& that) = delete;

    dispatch_queue_t notificationQ;
    xpc_connection_t service;
};

} //namespace SecTranslocate
} //namespace Security

#endif /* SecTranslocateXPCServer_hpp */
