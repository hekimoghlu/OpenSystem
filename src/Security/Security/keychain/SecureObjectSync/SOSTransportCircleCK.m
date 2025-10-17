/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
//  SOSTransportCircleCK.m
//  Security
//
//
//

#import <Foundation/Foundation.h>
#import "keychain/SecureObjectSync/SOSTransport.h"
#import "keychain/SecureObjectSync/SOSAccountPriv.h"
#import "SOSTransportCircleCK.h"

@implementation SOSCKCircleStorage

-(id) init
{
    if ((self = [super init])) {
        SOSRegisterTransportCircle(self);
    }
    return self;
}

-(id) initWithAccount:(SOSAccount*)acct
{
    if ((self = [super init])) {
        self.account = acct;
    }
    return self;
}

-(CFIndex) getTransportType
{
    return kCK;
}
-(SOSAccount*) getAccount
{
    return self.account;
}

-(bool) expireRetirementRecords:(CFDictionaryRef) retirements err:(CFErrorRef *)error
{
    return true;
}

-(bool) flushChanges:(CFErrorRef *)error
{
    return true;
}
-(bool) postCircle:(CFStringRef)circleName circleData:(CFDataRef)circle_data err:(CFErrorRef *)error
{
    return true;
}
-(bool) postRetirement:(CFStringRef) circleName peer:(SOSPeerInfoRef) peer err:(CFErrorRef *)error
{
    return true;
}

-(CFDictionaryRef)handleRetirementMessages:(CFMutableDictionaryRef) circle_retirement_messages_table err:(CFErrorRef *)error
{
    return NULL;
}
-(CFArrayRef)CF_RETURNS_RETAINED handleCircleMessagesAndReturnHandledCopy:(CFMutableDictionaryRef) circle_circle_messages_table err:(CFErrorRef *)error
{
    return NULL;
}

@end
