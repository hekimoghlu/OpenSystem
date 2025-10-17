/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
//  msdosProgressHelper.m
//  msdos.appex
//
//  Created by William Stouder-Studenmund on 5/1/24.
//

#import "msdosProgressHelper.h"
#import <FSKit/FSKit.h>

@implementation msdosProgressHelper

-(nullable instancetype)initWithProgress:(NSProgress *)progress
{
    self = [super init];
    if (self) {
        _parentProgress = progress;
        _childProgress = nil;
    }
    return self;
}

-(NSError* _Nullable)startPhase:(NSString *)description
                parentUnitCount:(int64_t)parentUnitCount
                phaseTotalCount:(int64_t)phaseTotalCount
               completedCounter:(const unsigned int *)completedCounter
{
    if (_childProgress != nil) {
        // We are in the middle a phase - expect it to end before starting a new one.
        os_log_fault(OS_LOG_DEFAULT, "%s missing endPhase call for %@", __FUNCTION__, _parentProgress.localizedDescription);
        return fs_errorForPOSIXError(EINVAL);
    }

    _parentProgress.localizedDescription = description;
    _childProgress = [NSProgress progressWithTotalUnitCount:phaseTotalCount];
    [_parentProgress addChild:_childProgress withPendingUnitCount:parentUnitCount];

    return nil;
}

-(void)endPhase:(NSString *)description
{
    // We expect to be called for cleanup once sometime after startPhase got called.
    // Silently do nothing if we got unexpected end call
    if (_childProgress) {
        _parentProgress.localizedDescription = description;
        _childProgress.completedUnitCount = _childProgress.totalUnitCount;
        _childProgress = nil;
    }
}

@end
