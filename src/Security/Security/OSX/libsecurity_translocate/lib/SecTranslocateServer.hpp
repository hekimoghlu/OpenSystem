/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
/* Purpose: This header defines the Translocation "Server" it houses the XPC Listener, Disk Arbitration Monitor,
    Launch Services Notification monitor, and interfaces for the process housing the server to request translocation
    services.
 */

#ifndef SecTranslocateServer_hpp
#define SecTranslocateServer_hpp

#include <string>
#include <dispatch/dispatch.h>

#include "SecTranslocateInterface.hpp"
#include "SecTranslocateUtilities.hpp"
#include "SecTranslocateLSNotification.hpp"
#include "SecTranslocateDANotification.hpp"
#include "SecTranslocateXPCServer.hpp"


namespace Security {
    
namespace SecTranslocate {

using namespace std;

class TranslocatorServer: public Translocator
{
public:
    TranslocatorServer(dispatch_queue_t q);
    ~TranslocatorServer();

    string translocatePathForUser(const TranslocationPath &originalPath, ExtendedAutoFileDesc &destPath) override;
    string translocatePathForUser(const GenericTranslocationPath &originalPath, ExtendedAutoFileDesc &destPath) override;
    bool destroyTranslocatedPathForUser(const string &translocatedPath) override;
    void appLaunchCheckin(pid_t pid) override;
    
private:
    TranslocatorServer() = delete;
    TranslocatorServer(const TranslocatorServer &that) = delete;
    dispatch_queue_t syncQ;
    DANotificationMonitor da;
    LSNotificationMonitor ls;
    XPCServer xpc;
    dispatch_source_t cleanupTimer;

    void setupPeriodicCleanup();
};
    
} //namespace SecTranslocate
} //namespace Security

#endif /* SecTranslocateServer_hpp */
