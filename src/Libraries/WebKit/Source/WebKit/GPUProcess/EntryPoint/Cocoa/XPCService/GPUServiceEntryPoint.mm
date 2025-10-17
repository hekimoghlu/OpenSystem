/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
#import "GPUProcess.h"
#import "WKBase.h"
#import "XPCServiceEntryPoint.h"

#if ENABLE(GPU_PROCESS)

namespace WebKit {

class GPUServiceInitializerDelegate : public XPCServiceInitializerDelegate {
public:
    GPUServiceInitializerDelegate(OSObjectPtr<xpc_connection_t> connection, xpc_object_t initializerMessage)
        : XPCServiceInitializerDelegate(WTFMove(connection), initializerMessage)
    {
    }
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)

extern "C" WK_EXPORT void GPU_SERVICE_INITIALIZER(xpc_connection_t connection, xpc_object_t initializerMessage);

void GPU_SERVICE_INITIALIZER(xpc_connection_t connection, xpc_object_t initializerMessage)
{
    g_jscConfig.vmCreationDisallowed = true;
    g_jscConfig.vmEntryDisallowed = true;
    g_wtfConfig.useSpecialAbortForExtraSecurityImplications = true;

    WTF::initializeMainThread();
    {
        JSC::Options::initialize();
        JSC::Options::AllowUnfinalizedAccessScope scope;
        JSC::ExecutableAllocator::disableJIT();
        JSC::Options::useWasm() = false;
        JSC::Options::notifyOptionsChanged();
    }
    WTF::compilerFence();

#if ENABLE(GPU_PROCESS)
    WebKit::XPCServiceInitializer<WebKit::GPUProcess, WebKit::GPUServiceInitializerDelegate>(connection, initializerMessage);
#endif // ENABLE(GPU_PROCESS)

    JSC::Config::finalize();
}
