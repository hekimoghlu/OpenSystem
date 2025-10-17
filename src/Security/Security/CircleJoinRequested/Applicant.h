/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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
//  Applicant.h
//  Security
//
//  Created by J Osborne on 3/7/13.
//  Copyright (c) 2013 Apple Inc. All Rights Reserved.
//

#import <Foundation/Foundation.h>
#include "keychain/SecureObjectSync/SOSPeerInfo.h"

typedef enum {
    ApplicantWaiting,
    ApplicantOnScreen,
    ApplicantRejected,
    ApplicantAccepted,
} ApplicantUIState;

@interface Applicant : NSObject
@property (readwrite) ApplicantUIState applicantUIState;
@property (readonly) NSString *applicantUIStateName;
@property (readwrite) SOSPeerInfoRef rawPeerInfo;
@property (readonly) NSString *name;
@property (readonly) NSString *idString;
@property (readonly) NSString *deviceType;
-(id)initWithPeerInfo:(SOSPeerInfoRef) peerInfo;
-(NSString *)description;
-(void)dealloc;
@end
