/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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
/* Purpose:
    This header defines the client side interface (xpc client for translocation)
 */

#ifndef SecTranslocateClient_hpp
#define SecTranslocateClient_hpp

#include <string>
#include <dispatch/dispatch.h>

#include "SecTranslocateInterface.hpp"
#include "SecTranslocateShared.hpp"

namespace Security {

namespace SecTranslocate {

using namespace std;

class TranslocatorClient: public Translocator
{
public:
    TranslocatorClient(dispatch_queue_t q);
    ~TranslocatorClient();

    string translocatePathForUser(const TranslocationPath &originalPath, ExtendedAutoFileDesc &destFd) override;
    string translocatePathForUser(const GenericTranslocationPath &originalPath, ExtendedAutoFileDesc &destFd) override;
    bool destroyTranslocatedPathForUser(const string &translocatedPath) override;
    void appLaunchCheckin(pid_t pid) override;

private:
    TranslocatorClient() = delete;
    TranslocatorClient(const TranslocatorClient &that) = delete;
    
    string requestTranslocation(const int fdToTranslocate, const char * pathInsideTranslocationPoint, const int destFd, const TranslocationOptions flags);
    
    dispatch_queue_t syncQ;
    xpc_connection_t service;
};

} //namespace SecTranslocate
} //namespace Security

#endif /* SecTranslocateClient_hpp */
