/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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

#import <TargetConditionals.h>
#import <Foundation/Foundation.h>
#import <IDS/IDS.h>

#import "keychain/categories/NSError+UsefulConstructors.h"

#import "OTPairingPacketContext.h"
#import "OTPairingConstants.h"

@interface OTPairingPacketContext ()
@property (readwrite, atomic) NSDictionary *message;
@property (readwrite, atomic) NSString *fromID;
@property (readwrite, atomic) NSString *incomingResponseIdentifier;
@property (readwrite, atomic) NSString *outgoingResponseIdentifier;
@end

@implementation OTPairingPacketContext

@synthesize message = _message;
@synthesize fromID = _fromID;
@synthesize incomingResponseIdentifier = _incomingResponseIdentifier;
@synthesize outgoingResponseIdentifier = _outgoingResponseIdentifier;
@synthesize error = _error;

- (instancetype)initWithMessage:(NSDictionary *)message fromID:(NSString *)fromID context:(IDSMessageContext *)context
{
    if ((self = [super init])) {
        self.message = message;
        self.fromID = fromID;
        self.incomingResponseIdentifier = context.incomingResponseIdentifier;
        self.outgoingResponseIdentifier = context.outgoingResponseIdentifier;
    }
    return self;
}

- (enum OTPairingIDSMessageType)messageType
{
    NSNumber *typeNum;
    enum OTPairingIDSMessageType type;

    typeNum = self.message[OTPairingIDSKeyMessageType];
    if (typeNum != nil) {
        type = [typeNum intValue];
    } else {
        // From older internal builds; remove soon
        if (self.packetData != nil) {
            type = OTPairingIDSMessageTypePacket;
        } else {
            type = OTPairingIDSMessageTypeError;
        }
    }

    return type;
}

- (NSString *)sessionIdentifier
{
    return self.message[OTPairingIDSKeySession];
}

- (NSData *)packetData
{
    return self.message[OTPairingIDSKeyPacket];
}

- (NSError *)error
{
    if (self.messageType != OTPairingIDSMessageTypeError) {
        return nil;
    }

    if (!self->_error) {
        NSString *errorString = self.message[OTPairingIDSKeyErrorDescription];
        self->_error = [NSError errorWithDomain:OTPairingErrorDomain code:OTPairingErrorTypeRemote description:errorString];
    }

    return self->_error;
}

@end
