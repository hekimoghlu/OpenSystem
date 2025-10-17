/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 3, 2025.
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
#include "config.h"
#include "ScopedRenderingResourcesRequest.h"

#if PLATFORM(COCOA)

#import <Metal/Metal.h>
#import <objc/NSObjCRuntime.h>
#import <pal/spi/cocoa/MetalSPI.h>
#import <wtf/BlockObjCExceptions.h>
#import <wtf/RetainPtr.h>
#import <wtf/RunLoop.h>
#import <wtf/Seconds.h>

OBJC_PROTOCOL(MTLDevice);
OBJC_PROTOCOL(MTLDeviceSPI);

namespace WebKit {

static constexpr Seconds freeRenderingResourcesTimeout = 1_s;
static bool didScheduleFreeRenderingResources;

void ScopedRenderingResourcesRequest::scheduleFreeRenderingResources()
{
    if (didScheduleFreeRenderingResources)
        return;
    RunLoop::protectedMain()->dispatchAfter(freeRenderingResourcesTimeout, freeRenderingResources);
    didScheduleFreeRenderingResources = true;
}

void ScopedRenderingResourcesRequest::freeRenderingResources()
{
    didScheduleFreeRenderingResources = false;
    if (s_requests)
        return;
    BEGIN_BLOCK_OBJC_EXCEPTIONS
#if PLATFORM(MAC)
    auto devices = adoptNS(MTLCopyAllDevices());
    for (id<MTLDevice> device : devices.get()) {
        if ([device respondsToSelector:@selector(_purgeDevice)])
            [(_MTLDevice *)device _purgeDevice];
    }
#else
    RetainPtr<MTLDevice> devicePtr = adoptNS(MTLCreateSystemDefaultDevice());
    if ([devicePtr.get() respondsToSelector:@selector(_purgeDevice)])
        [(_MTLDevice *)devicePtr.get() _purgeDevice];
#endif
    END_BLOCK_OBJC_EXCEPTIONS
}

}

#endif
