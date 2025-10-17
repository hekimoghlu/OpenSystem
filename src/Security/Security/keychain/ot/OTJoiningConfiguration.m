/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
#import "keychain/ot/OTJoiningConfiguration.h"

#if __OBJC2__

NS_ASSUME_NONNULL_BEGIN

@implementation OTJoiningConfiguration

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (instancetype)initWithProtocolType:(NSString*)protocolType
                      uniqueDeviceID:(NSString*)uniqueDeviceID
                      uniqueClientID:(NSString*)uniqueClientID
                         pairingUUID:(NSString* _Nullable)pairingUUID
                               epoch:(uint64_t)epoch
                         isInitiator:(BOOL)isInitiator
{
    if ((self = [super init])) {
        self.protocolType = protocolType;
        self.uniqueDeviceID = uniqueDeviceID;
        self.uniqueClientID = uniqueClientID;
        self.isInitiator = isInitiator;
        self.pairingUUID = pairingUUID;
        self.epoch = epoch;
        self.testsEnabled = NO;

        _timeout = 0;
    }
    return self;
}

- (void)encodeWithCoder:(nonnull NSCoder *)coder { 
    [coder encodeObject:_protocolType forKey:@"protocolType"];
    [coder encodeObject:_uniqueClientID forKey:@"uniqueClientID"];
    [coder encodeObject:_uniqueDeviceID forKey:@"uniqueDeviceID"];
    [coder encodeBool:_isInitiator forKey:@"isInitiator"];
    [coder encodeObject:_pairingUUID forKey:@"pairingUUID"];
    [coder encodeInt64:_epoch forKey:@"epoch"];
    [coder encodeInt64:_timeout forKey:@"timeout"];
    [coder encodeBool:_testsEnabled forKey:@"testsEnabled"];
}

- (nullable instancetype)initWithCoder:(nonnull NSCoder *)decoder {
    if ((self = [super init])) {
        _protocolType = [decoder decodeObjectOfClass:[NSString class] forKey:@"protocolType"];
        _uniqueClientID = [decoder decodeObjectOfClass:[NSString class] forKey:@"uniqueClientID"];
        _uniqueDeviceID = [decoder decodeObjectOfClass:[NSString class] forKey:@"uniqueDeviceID"];
        _isInitiator = [decoder decodeBoolForKey:@"isInitiator"];
        _pairingUUID = [decoder decodeObjectOfClass:[NSString class] forKey:@"pairingUUID"];
        _epoch = [decoder decodeInt64ForKey:@"epoch"];
        _timeout = [decoder decodeInt64ForKey:@"timeout"];
        _testsEnabled = [decoder decodeBoolForKey:@"testsEnabled"];
    }
    return self;
}

- (void)enableForTests
{
    self.testsEnabled = YES;
}

@end
NS_ASSUME_NONNULL_END

#endif /* __OBJC2__ */
