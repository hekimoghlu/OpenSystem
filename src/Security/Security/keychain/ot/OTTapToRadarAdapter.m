/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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

#import "OTTapToRadarAdapter.h"
#import "utilities/debugging.h"

#import <TapToRadarKit/TapToRadarKit.h>
#import <os/feature_private.h>

@implementation OTTapToRadarActualAdapter

- (id)init {
    if((self = [super init])) {

    }
    return self;
}

- (void)postHomePodLostTrustTTR:(NSString*)identifiers {
    if([TapToRadarService class] == nil) {
        secnotice("octagon-ttr", "Trust lost, but TTR service not available");
        return;
    }

    if(!os_feature_enabled(Security, TTRTrustLossOnHomePod)) {
        secnotice("octagon-ttr", "Trust lost, not posting TTR due to feature flag");
        return;
    }

    secnotice("octagon-ttr", "Trust lost, posting TTR");

    RadarDraft* draft = [[RadarDraft alloc] init];
    draft.component = [[RadarComponent alloc] initWithName:@"Security" version:@"iCloud Keychain" identifier:606179];
    draft.isUserInitiated = NO;
    draft.reproducibility = ReproducibilityNotApplicable;
    draft.remoteDeviceClasses = RemoteDeviceClassesiPhone |
    RemoteDeviceClassesiPad |
    RemoteDeviceClassesMac |
    RemoteDeviceClassesAppleWatch |
    RemoteDeviceClassesAppleTV |
    RemoteDeviceClassesHomePod;
    draft.remoteDeviceSelections = RemoteDeviceSelectionsHomeKitHome;
    draft.title = @"Lost CDP trust";

    draft.problemDescription = [NSString stringWithFormat:@"HomePod unexpectedly lost CDP trust (please do not file this radar if you performed Reset Protected Data on another device, or otherwise intended to cause CDP trust loss on this HomePod). To disable this prompt for testing, turn off the Security/TTRTrustLossOnHomePod feature flag on the HomePod.\n\n%@", identifiers];
    draft.classification = ClassificationOtherBug;

    TapToRadarService* s = [TapToRadarService shared];
    [s createDraft:draft forProcessNamed:@"CDP" withDisplayReason:@"HomePod lost CDP/Manatee access" completionHandler:^(NSError * _Nullable error) {
        if(error == nil) {
            secnotice("octagon", "Created TTR successfully");
        } else {
            secnotice("octagon", "Created TTR with error: %@", error);
        }
    }];
}

@end

#endif // OCTAGON
