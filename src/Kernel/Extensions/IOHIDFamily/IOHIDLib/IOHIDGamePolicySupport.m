/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
#include <dlfcn.h>
#include "IOHIDGamePolicySupport.h"
#include <Foundation/Foundation.h>
#include <dispatch/dispatch.h>

#define GamePolicyFrameworkPath "/System/Library/PrivateFrameworks/GamePolicy.framework/GamePolicy"

#define GPMonitorClassName "GPProcessMonitor"

@class GPProcessInfo;
@class GPProcessMonitor;

@protocol GPMonitorInfoProtocol <NSObject>

typedef void(^GPProcessInfoUpdateHandler)(GPProcessMonitor *monitor, GPProcessInfo *processInfo);

- (void)setUpdateHandler:(nullable GPProcessInfoUpdateHandler)block;

@end

static void *__loadFramework()
{
 
    static void *gpHandle = NULL;
    static dispatch_once_t gpOnce = 0;

    dispatch_once(&gpOnce, ^{
        gpHandle = dlopen(GamePolicyFrameworkPath, RTLD_LAZY);

        if (!gpHandle) {
            return;
        }
    });

    return gpHandle;
}

IOReturn IOHIDAnalyticsGetConsoleModeStatus(ConsoleModeBlock replyBlock)
{

    IOReturn returnStatus = kIOReturnError;

    void * gpHandle = __loadFramework();

    if (!gpHandle) {
        return kIOReturnError;
    }

    Class GPMonitor_class = NSClassFromString(@GPMonitorClassName);
    if (GPMonitor_class) {
        SEL montiorForCurrentProcess_sel = sel_getUid("monitorForCurrentProcess");
        id<GPMonitorInfoProtocol> gameMonitor = [GPMonitor_class performSelector:montiorForCurrentProcess_sel];
        if (!gameMonitor) {
            return kIOReturnError;
        }

        [gameMonitor setUpdateHandler:^(GPProcessMonitor * __unused monitor, GPProcessInfo *processInfo) {
            id processInfoID = processInfo;
            SEL isIdentifiedGame_sel = sel_getUid("isIdentifiedGame");
            BOOL nameStatus = (BOOL)[processInfoID performSelector:isIdentifiedGame_sel];
            replyBlock(nameStatus == YES);
        }];
        returnStatus = kIOReturnSuccess;
    }

    return returnStatus;
}
