/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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
/* Purpose: This header defines the generic Translocator interface, implemented by the client and server,
    and the Translocator factory method to make a client or server object
 */

#ifndef SecTranslocateInterface_h
#define SecTranslocateInterface_h

#include <string>
#include <unistd.h>

#include "SecTranslocateShared.hpp"

namespace Security {
namespace SecTranslocate {

using namespace std;

#define SECTRANSLOCATE_XPC_SERVICE_NAME "com.apple.security.translocation"

class Translocator
{
public:
    virtual ~Translocator() {};
    virtual string translocatePathForUser(const TranslocationPath &originalPath, ExtendedAutoFileDesc &destFd) = 0;
    virtual string translocatePathForUser(const GenericTranslocationPath &originalPath, ExtendedAutoFileDesc &destFd) = 0;
    virtual bool destroyTranslocatedPathForUser(const string &translocatedPath) = 0;
    virtual void appLaunchCheckin(pid_t pid) = 0;
};

Translocator* getTranslocator(bool isServer=false);

} //namespace SecTranslocate
} //namespace Security
#endif /* SecTranslocateInterface_h */
