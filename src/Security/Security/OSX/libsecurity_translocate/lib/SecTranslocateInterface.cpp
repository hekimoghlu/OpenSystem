/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
#include <exception>
#include <dispatch/dispatch.h>

#include <security_utilities/unix++.h>
#include <security_utilities/logging.h>

#include "SecTranslocateInterface.hpp"
#include "SecTranslocateServer.hpp"
#include "SecTranslocateClient.hpp"

namespace Security {
namespace SecTranslocate {

using namespace std;

Translocator* getTranslocator(bool isServer)
{
    static dispatch_once_t initialized;
    static Translocator* me = NULL;
    static dispatch_queue_t q;
    __block exception_ptr exception(0);

    if(isServer && me)
    {
        Syslog::critical("SecTranslocate: getTranslocator, asked for server but previously intialized as client");
        UnixError::throwMe(EINVAL);
    }

    dispatch_once(&initialized, ^{
        try
        {
            q = dispatch_queue_create(isServer?"com.apple.security.translocate":"com.apple.security.translocate-client", DISPATCH_QUEUE_SERIAL);
            if(q == NULL)
            {
                Syslog::critical("SecTranslocate: getTranslocator, failed to create queue");
                UnixError::throwMe(ENOMEM);
            }

            if(isServer)
            {
                me = new TranslocatorServer(q);
            }
            else
            {
                me = new TranslocatorClient(q);
            }
        }
        catch (...)
        {
            Syslog::critical("SecTranslocate: error while creating Translocator");
            exception = current_exception();
        }
    });

    if (me == NULL)
    {
        if (exception)
        {
            rethrow_exception(exception); //we already logged in this case.
        }
        else
        {
            Syslog::critical("SecTranslocate: Translocator initialization failed");
            UnixError::throwMe(EINVAL);
        }
    }

    return me;
}

} //namespace SecTranslocate
} //namespace Security
