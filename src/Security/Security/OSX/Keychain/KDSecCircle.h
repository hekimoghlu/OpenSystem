/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
#import <Security/SecureObjectSync/SOSCloudCircle.h>
#import <Foundation/Foundation.h>

@interface KDSecCircle : NSObject

@property (readonly) BOOL isInCircle;
@property (readonly) BOOL isOutOfCircle;

@property (readonly) SOSCCStatus rawStatus;

@property (readonly) NSString *status;
@property (readonly) NSError *error;

// Both of these are arrays of KDCircelPeer objects
@property (readonly) NSArray *peers;
@property (readonly) NSArray *applicants;

-(void)addChangeCallback:(dispatch_block_t)callback;
-(id)init;

// these are "try to", and may (most likely will) not complete by the time they return
-(void)enableSync;
-(void)disableSync;
-(void)rejectApplicantId:(NSString*)applicantId;
-(void)acceptApplicantId:(NSString*)applicantId;

@end
