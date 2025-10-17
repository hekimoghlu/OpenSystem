/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
#import "WKBase.h"
#import "WebProcess.h"
#import "XPCServiceEntryPoint.h"

#if USE(TZONE_MALLOC)
#import <bmalloc/TZoneHeapManager.h>
#endif

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WebCoreThreadSystemInterface.h>
#import <pal/spi/ios/GraphicsServicesSPI.h>
#endif

extern "C" WK_EXPORT void WEBCONTENT_SERVICE_INITIALIZER(xpc_connection_t connection, xpc_object_t initializerMessage);

void WEBCONTENT_SERVICE_INITIALIZER(xpc_connection_t connection, xpc_object_t initializerMessage)
{
#if USE(TZONE_MALLOC)
    bmalloc::api::TZoneHeapManager::setBucketParams(4);
#endif
    WTF::initializeMainThread();

    // Remove the WebProcessShim from the DYLD_INSERT_LIBRARIES environment variable so any processes spawned by
    // the this process don't try to insert the shim and crash.
    WebKit::EnvironmentUtilities::removeValuesEndingWith("DYLD_INSERT_LIBRARIES", "/WebProcessShim.dylib");

#if PLATFORM(IOS_FAMILY)
    GSInitialize();
    InitWebCoreThreadSystemInterface();
#endif // PLATFORM(IOS_FAMILY)

    WebKit::XPCServiceInitializer<WebKit::WebProcess, WebKit::XPCServiceInitializerDelegate, true>(connection, initializerMessage);
}
