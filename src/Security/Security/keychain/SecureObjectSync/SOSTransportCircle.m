/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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

#import "keychain/SecureObjectSync/SOSAccountPriv.h"
#include "keychain/SecureObjectSync/SOSTransport.h"

@implementation SOSCircleStorageTransport

@synthesize account = account;

-(id)init
{
    return [super init];

}
-(SOSCircleStorageTransport*) initWithAccount:(SOSAccount*)acct
{
    if ((self = [super init])) {
        self.account = acct;
    }
    return self;
}

-(SOSAccount*)getAccount
{
    return self.account;
}

-(CFIndex)circleGetTypeID
{
    return kUnknown;
}
-(CFIndex)getTransportType
{
    return kUnknown;
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

-(bool) postRetirement:(CFStringRef) circleName peer:(SOSPeerInfoRef) peer err:(CFErrorRef *)error{
    return true;
}

-(CFDictionaryRef)handleRetirementMessages:(CFMutableDictionaryRef) circle_retirement_messages_table err:(CFErrorRef *)error
{
    return CFDictionaryCreateMutableForCFTypes(kCFAllocatorDefault);
}

-(CFArrayRef) handleCircleMessagesAndReturnHandledCopy:(CFMutableDictionaryRef) circle_circle_messages_table err:(CFErrorRef *)error
{
    return CFArrayCreateMutableForCFTypes(kCFAllocatorDefault);
}

@end

