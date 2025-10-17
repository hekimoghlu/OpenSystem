/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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

//
//  Applicant.m
//  Security
//
//  Created by J Osborne on 3/7/13.
//  Copyright (c) 2013 Apple Inc. All Rights Reserved.
//

#import "Applicant.h"
#include <utilities/SecCFRelease.h>

@implementation Applicant

-(id)initWithPeerInfo:(SOSPeerInfoRef)peerInfo
{
    if ((self = [super init])) {
        self.rawPeerInfo = CFRetainSafe(peerInfo);
        self.applicantUIState = ApplicantWaiting;
    }    
    return self;
}

-(NSString*)idString
{
    return (__bridge NSString *)(SOSPeerInfoGetPeerID(self.rawPeerInfo));
}

-(NSString *)name
{
    return (__bridge NSString *)(SOSPeerInfoGetPeerName(self.rawPeerInfo));
}

-(void)dealloc
{
	if (self->_rawPeerInfo) {
		CFRelease(self->_rawPeerInfo);
	}
}

-(NSString *)description
{
	return [NSString stringWithFormat:@"%@=%@", self.rawPeerInfo, self.applicantUIStateName];
}

-(NSString *)deviceType
{
    return (__bridge NSString *)(SOSPeerInfoGetPeerDeviceType(self.rawPeerInfo));
}

-(NSString *)applicantUIStateName
{
	switch (self.applicantUIState) {
		case ApplicantWaiting:
			return @"Waiting";

		case ApplicantOnScreen:
			return @"OnScreen";

		case ApplicantRejected:
			return @"Rejected";

		case ApplicantAccepted:
			return @"Accepted";

		default:
			return [NSString stringWithFormat:@"UnknownState#%d", self.applicantUIState];
	}
}

@end
