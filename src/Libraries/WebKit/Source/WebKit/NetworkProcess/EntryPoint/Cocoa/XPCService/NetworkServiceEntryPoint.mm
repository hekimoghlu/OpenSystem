/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#import "config.h"

#import "EnvironmentUtilities.h"
#import "NetworkProcess.h"
#import "WKBase.h"
#import "XPCServiceEntryPoint.h"

namespace WebKit {

class NetworkServiceInitializerDelegate : public XPCServiceInitializerDelegate {
public:
    NetworkServiceInitializerDelegate(OSObjectPtr<xpc_connection_t> connection, xpc_object_t initializerMessage)
        : XPCServiceInitializerDelegate(WTFMove(connection), initializerMessage)
    {
    }
};

template<>
void initializeAuxiliaryProcess<NetworkProcess>(AuxiliaryProcessInitializationParameters&& parameters)
{
    static NeverDestroyed<NetworkProcess> networkProcess(WTFMove(parameters));
}

} // namespace WebKit

using namespace WebKit;

extern "C" WK_EXPORT void NETWORK_SERVICE_INITIALIZER(xpc_connection_t connection, xpc_object_t initializerMessage);

void NETWORK_SERVICE_INITIALIZER(xpc_connection_t connection, xpc_object_t initializerMessage)
{
    WTF::initializeMainThread();
    XPCServiceInitializer<NetworkProcess, NetworkServiceInitializerDelegate>(connection, initializerMessage);
}
