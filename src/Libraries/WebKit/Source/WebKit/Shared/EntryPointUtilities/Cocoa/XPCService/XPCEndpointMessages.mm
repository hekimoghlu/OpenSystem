/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
#import "XPCEndpointMessages.h"

#import "GPUConnectionToWebProcess.h"
#import "GPUProcess.h"
#import "LaunchServicesDatabaseManager.h"
#import "LaunchServicesDatabaseXPCConstants.h"
#import "RemoteMediaPlayerManagerProxy.h"
#import "VideoReceiverEndpointMessage.h"
#import "XPCEndpoint.h"
#import <wtf/RunLoop.h>
#import <wtf/RuntimeApplicationChecks.h>
#import <wtf/text/WTFString.h>

namespace WebKit {

#if HAVE(LSDATABASECONTEXT)
static void handleLaunchServiceDatabaseMessage(xpc_object_t message)
{
    auto xpcEndPoint = xpc_dictionary_get_value(message, LaunchServicesDatabaseXPCConstants::xpcLaunchServicesDatabaseXPCEndpointNameKey);
    if (!xpcEndPoint || xpc_get_type(xpcEndPoint) != XPC_TYPE_ENDPOINT)
        return;

    LaunchServicesDatabaseManager::singleton().setEndpoint(xpcEndPoint);
}
#endif

#if ENABLE(LINEAR_MEDIA_PLAYER)
static void handleVideoReceiverEndpointMessage(xpc_object_t message)
{
    ASSERT(isMainRunLoop());
    RELEASE_ASSERT(isInGPUProcess());

    auto endpointMessage = VideoReceiverEndpointMessage::decode(message);
    if (!endpointMessage.processIdentifier())
        return;

    if (RefPtr webProcessConnection = GPUProcess::singleton().webProcessConnection(*endpointMessage.processIdentifier()))
        webProcessConnection->remoteMediaPlayerManagerProxy().handleVideoReceiverEndpointMessage(endpointMessage);
}

static void handleVideoReceiverSwapEndpointsMessage(xpc_object_t message)
{
    ASSERT(isMainRunLoop());
    RELEASE_ASSERT(isInGPUProcess());

    auto endpointMessage = VideoReceiverSwapEndpointsMessage::decode(message);
    if (!endpointMessage.processIdentifier())
        return;

    if (RefPtr webProcessConnection = GPUProcess::singleton().webProcessConnection(*endpointMessage.processIdentifier()))
        webProcessConnection->remoteMediaPlayerManagerProxy().handleVideoReceiverSwapEndpointsMessage(endpointMessage);
}
#endif

void handleXPCEndpointMessage(xpc_object_t message, const String& messageName)
{
    ASSERT_UNUSED(messageName, messageName);
    RELEASE_ASSERT(xpc_get_type(message) == XPC_TYPE_DICTIONARY);

#if HAVE(LSDATABASECONTEXT)
    if (messageName == LaunchServicesDatabaseXPCConstants::xpcLaunchServicesDatabaseXPCEndpointMessageName) {
        handleLaunchServiceDatabaseMessage(message);
        return;
    }
#endif

#if ENABLE(LINEAR_MEDIA_PLAYER)
    if (messageName == VideoReceiverEndpointMessage::messageName()) {
        RunLoop::main().dispatch([message = OSObjectPtr(message)] {
            handleVideoReceiverEndpointMessage(message.get());
        });
        return;
    }

    if (messageName == VideoReceiverSwapEndpointsMessage::messageName()) {
        RunLoop::main().dispatch([message = OSObjectPtr(message)] {
            handleVideoReceiverSwapEndpointsMessage(message.get());
        });
        return;
    }
#endif

}

} // namespace WebKit
