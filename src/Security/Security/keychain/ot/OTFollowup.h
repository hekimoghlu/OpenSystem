/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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
#ifndef _OT_FOLLOWUP_H_
#define _OT_FOLLOWUP_H_

#if OCTAGON

#import <Foundation/Foundation.h>
#import <CoreCDP/CDPFollowUpController.h>
#import "keychain/ckks/CKKSAnalytics.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperSpecificUser.h"

#if TARGET_OS_IOS || TARGET_OS_OSX
#define OCTAGON_PLATFORM_SUPPORTS_RK_CFU 1
#else
#define OCTAGON_PLATFORM_SUPPORTS_RK_CFU 0
#endif

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(uint8_t, OTFollowupContextType) {
    OTFollowupContextTypeNone = 0,
#if OCTAGON_PLATFORM_SUPPORTS_RK_CFU
    OTFollowupContextTypeRecoveryKeyRepair = 1,
#endif
    OTFollowupContextTypeStateRepair = 2,
    OTFollowupContextTypeConfirmExistingSecret = 3,
    OTFollowupContextTypeSecureTerms = 4,
};
NSString* OTFollowupContextTypeToString(OTFollowupContextType contextType);

@protocol OctagonFollowUpControllerProtocol
- (BOOL)postFollowUpWithContext:(CDPFollowUpContext *)context error:(NSError **)error;
- (BOOL)clearFollowUpWithContext:(CDPFollowUpContext *)context error:(NSError **)error;
@end
@interface CDPFollowUpController (Octagon) <OctagonFollowUpControllerProtocol>
@end

@interface OTFollowup : NSObject
- (id)initWithFollowupController:(id<OctagonFollowUpControllerProtocol>)cdpFollowupController;

- (BOOL)postFollowUp:(OTFollowupContextType)contextType
       activeAccount:(TPSpecificUser*)activeAccount
               error:(NSError **)error;
- (BOOL)clearFollowUp:(OTFollowupContextType)contextType
        activeAccount:(TPSpecificUser*)activeAccount
                error:(NSError **)error;

- (NSDictionary *_Nullable)sysdiagnoseStatus;
- (NSDictionary<NSString*,NSNumber*> *)sfaStatus;
@end

@interface OTFollowup (Testing)
// Reports on whether this individual OTFollowUp object has posted a CFU of this type.
- (BOOL)hasPosted:(OTFollowupContextType)contextType;
- (void)clearAllPostedFlags;
@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON

#endif // _OT_FOLLOWUP_H_
