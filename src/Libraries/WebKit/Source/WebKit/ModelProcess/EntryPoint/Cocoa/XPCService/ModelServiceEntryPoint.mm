/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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
#import "ModelProcess.h"
#import "WKBase.h"
#import "XPCServiceEntryPoint.h"

#if ENABLE(MODEL_PROCESS)

namespace WebKit {

class ModelServiceInitializerDelegate : public XPCServiceInitializerDelegate {
public:
    ModelServiceInitializerDelegate(OSObjectPtr<xpc_connection_t> connection, xpc_object_t initializerMessage)
        : XPCServiceInitializerDelegate(WTFMove(connection), initializerMessage)
    {
    }
};

template<>
void initializeAuxiliaryProcess<ModelProcess>(AuxiliaryProcessInitializationParameters&& parameters)
{
    static NeverDestroyed<ModelProcess> modelProcess(WTFMove(parameters));
}

} // namespace WebKit

#endif // ENABLE(MODEL_PROCESS)

extern "C" WK_EXPORT void MODEL_SERVICE_INITIALIZER(xpc_connection_t connection, xpc_object_t initializerMessage);

void MODEL_SERVICE_INITIALIZER(xpc_connection_t connection, xpc_object_t initializerMessage)
{
    WTF::initializeMainThread();

#if ENABLE(MODEL_PROCESS)
    WebKit::XPCServiceInitializer<WebKit::ModelProcess, WebKit::ModelServiceInitializerDelegate>(connection, initializerMessage);
#endif // ENABLE(MODEL_PROCESS)
}
