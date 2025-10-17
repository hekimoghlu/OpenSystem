/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#if OCTAGON

#import <utilities/debugging.h>

#import "keychain/ot/OTDeviceInformation.h"

#import "keychain/ot/ObjCImprovements.h"
#import <SystemConfiguration/SystemConfiguration.h>

#if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
#include <MobileGestalt.h>
#else
#include <AppleSystemInfo/AppleSystemInfo.h>
#endif

@implementation OTDeviceInformation

- (instancetype)initForContainerName:(NSString*)containerName
                           contextID:(NSString*)contextID
                               epoch:(uint64_t)epoch
                           machineID:(NSString* _Nullable)machineID
                             modelID:(NSString*)modelID
                          deviceName:(NSString*)deviceName
                        serialNumber:(NSString*)serialNumber
                           osVersion:(NSString*)osVersion
{
    if((self = [super init])) {
        //otcuttlefish context
        self.containerName = containerName;
        self.contextID = contextID;
        //our epoch
        self.epoch = epoch;

        self.machineID = machineID;
        self.modelID = modelID;
        self.deviceName = deviceName;
        self.serialNumber = serialNumber;
        self.osVersion = osVersion;
    }
    return self;
}

/// Returns True if the modelID is considered to be a Full Peer, False otherwise.
/// Defaults to false/limitedPeer if the modelID is unknown
+ (bool)isFullPeer:(NSString*)modelID {
    for (NSString* p in @[
        @"Mac",
        @"iPhone",
        @"iPad",
        @"iPod",
        @"Watch",
        @"RealityDevice",
    ]) {
        if ([modelID containsString:p]) {
            return true;
        }
    }
    return false;
}
@end

#endif // OCTAGON
