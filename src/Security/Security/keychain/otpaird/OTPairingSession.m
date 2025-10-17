/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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

#import <KeychainCircle/KeychainCircle.h>
#import <os/assumes.h>

#if !TARGET_OS_SIMULATOR
#import <MobileKeyBag/MobileKeyBag.h>
#endif /* !TARGET_OS_SIMULATOR */

#import "OTPairingSession.h"

@interface OTPairingSession ()
@property (readwrite) NSString *identifier;
@property (readwrite) KCPairingChannel *channel;
@property (readwrite) NSMutableArray<OTPairingCompletionHandler>* completionHandlers;
#if !TARGET_OS_SIMULATOR
@property MKBAssertionRef lockAssertion;
#endif /* !TARGET_OS_SIMULATOR */
@end

@implementation OTPairingSession

- (instancetype)initAsInitiator:(bool)initiator deviceInfo:(OTDeviceInformationActualAdapter *)deviceInfo identifier:(NSString *)identifier
{
    KCPairingChannelContext *channelContext = nil;

    if ((self = [super init])) {
        channelContext = [[KCPairingChannelContext alloc] init];
        channelContext.uniqueClientID = [NSUUID UUID].UUIDString;
        channelContext.uniqueDeviceID = [NSUUID UUID].UUIDString;
        channelContext.intent = KCPairingIntent_Type_SilentRepair;
        channelContext.model = deviceInfo.modelID;
        channelContext.osVersion = deviceInfo.osVersion;

        if (initiator) {
            os_assert(identifier == nil);
            self.identifier = [[NSUUID UUID] UUIDString];
            self.channel = [KCPairingChannel pairingChannelInitiator:channelContext];
        } else {
            os_assert(identifier != nil);
            self.identifier = identifier;
            self.channel = [KCPairingChannel pairingChannelAcceptor:channelContext];
        }
    }

    return self;
}

- (void)dealloc
{
#if !TARGET_OS_SIMULATOR
    if (self->_lockAssertion) {
        CFRelease(self->_lockAssertion);
        self->_lockAssertion = NULL;
    }
#endif /* !TARGET_OS_SIMULATOR */
}

#if !TARGET_OS_SIMULATOR
- (BOOL)acquireLockAssertion
{
    if (self->_lockAssertion == NULL) {
        CFErrorRef lockError = NULL;
        NSDictionary* lockOptions = @{
            (__bridge NSString *)kMKBAssertionTypeKey : (__bridge NSString *)kMKBAssertionTypeOther,
            (__bridge NSString *)kMKBAssertionTimeoutKey : @(60),
        };
        self->_lockAssertion = MKBDeviceLockAssertion((__bridge CFDictionaryRef)lockOptions, &lockError);

        if (self->_lockAssertion == NULL || lockError != NULL) {
            os_log(OS_LOG_DEFAULT, "Failed to obtain lock assertion: %@", lockError);
            if (lockError != NULL) {
                CFRelease(lockError);
            }
        }
    }

    return (self->_lockAssertion != NULL);
}
#endif /* !TARGET_OS_SIMULATOR */

- (void)addCompletionHandler:(OTPairingCompletionHandler)completionHandler
{
    if (self.completionHandlers == nil) {
        self.completionHandlers = [[NSMutableArray alloc] init];
    }
    [self.completionHandlers addObject:completionHandler];
}

- (void)didCompleteWithSuccess:(bool)success error:(NSError *)error
{
    for (OTPairingCompletionHandler completionHandler in self.completionHandlers) {
        completionHandler(success, error);
    }
}

@end
