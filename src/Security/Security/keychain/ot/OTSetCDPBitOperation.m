/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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
#if OCTAGON

#import "utilities/debugging.h"

#import "keychain/ot/OTSetCDPBitOperation.h"
#import "keychain/ot/ObjCImprovements.h"

@interface OTSetCDPBitOperation ()
@property OTOperationDependencies* deps;

@property NSOperation* finishedOp;
@end

@implementation OTSetCDPBitOperation
@synthesize intendedState = _intendedState;
@synthesize nextState = _nextState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
{
    if((self = [super init])) {
        _deps = dependencies;
        _intendedState = intendedState;
        _nextState = errorState;
    }
    return self;
}

- (void)groupStart
{
    NSError* localError = nil;
    [self.deps.stateHolder persistAccountChanges:^OTAccountMetadataClassC * _Nonnull(OTAccountMetadataClassC * _Nonnull metadata) {
        metadata.cdpState = OTAccountMetadataClassC_CDPState_ENABLED;
        return metadata;
    } error:&localError];

    if(localError) {
        secnotice("octagon-cdp-status", "Unable to set CDP bit: %@", localError);
    } else {
        secnotice("octagon-cdp-status", "Successfully set CDP bit");
        self.nextState = self.intendedState;
    }
}

@end

#endif // OCTAGON
