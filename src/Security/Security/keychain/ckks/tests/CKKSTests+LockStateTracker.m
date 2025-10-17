/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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

#import <XCTest/XCTest.h>
#import <OCMock/OCMock.h>

#import "keychain/ckks/CKKSLockStateTracker.h"
#import "keychain/ckks/tests/CKKSMockLockStateProvider.h"
#import "tests/secdmockaks/mockaks.h"

@interface CKKSTests_LockStateTracker : XCTestCase

@property CKKSMockLockStateProvider* lockStateProvider;
@property CKKSLockStateTracker* lockStateTracker;
@end

@implementation CKKSTests_LockStateTracker

- (void)setUp {
    [super setUp];

    self.lockStateProvider = [[CKKSMockLockStateProvider alloc] initWithCurrentLockStatus:NO];
    self.lockStateTracker = [[CKKSLockStateTracker alloc] initWithProvider:self.lockStateProvider];

    [SecMockAKS reset];
}

- (void)tearDown {
    self.lockStateProvider = nil;
    self.lockStateTracker = nil;
}

- (void)testLockedBehindOurBack {

    /*
     * check that we detect that lock errors force a recheck
     */

    NSError *lockError = [NSError errorWithDomain:NSOSStatusErrorDomain code:errSecInteractionNotAllowed userInfo:nil];
    NSError *fileError = [NSError errorWithDomain:NSOSStatusErrorDomain code:ENOENT userInfo:nil];

    XCTAssertFalse([self.lockStateTracker isLocked], "should start out unlocked");
    XCTAssertTrue([self.lockStateTracker isLockedError:lockError], "errSecInteractionNotAllowed is a lock errors");
    XCTAssertFalse([self.lockStateTracker isLocked], "should be unlocked after lock failure");

    XCTAssertFalse([self.lockStateTracker isLockedError:fileError], "file errors are not lock errors");
    XCTAssertFalse([self.lockStateTracker isLocked], "should be unlocked after lock failure");

    self.lockStateProvider.aksCurrentlyLocked = true;
    XCTAssertFalse([self.lockStateTracker isLocked], "should be reporting unlocked since we 'missed' the notification");

    XCTAssertFalse([self.lockStateTracker isLockedError:fileError], "file errors are not lock errors");
    XCTAssertFalse([self.lockStateTracker isLocked], "should be 'unlocked' after file errors");

    XCTAssertTrue([self.lockStateTracker isLockedError:lockError], "errSecInteractionNotAllowed is a lock errors");
    XCTAssertTrue([self.lockStateTracker isLocked], "should be locked after lock failure");

    self.lockStateProvider.aksCurrentlyLocked = false;
    [self.lockStateTracker recheck];

    XCTAssertFalse([self.lockStateTracker isLocked], "should be unlocked");
}

- (void)testWaitForUnlock {

    self.lockStateProvider.aksCurrentlyLocked = true;
    [self.lockStateTracker recheck];

    XCTestExpectation* expectation = [self expectationWithDescription: @"unlock occurs"];

    NSBlockOperation *unlockEvent = [NSBlockOperation blockOperationWithBlock:^{
        [expectation fulfill];
    }];
    [unlockEvent addDependency:[self.lockStateTracker unlockDependency]];
    NSOperationQueue *queue = [[NSOperationQueue alloc] init];

    [queue addOperation:unlockEvent];

    self.lockStateProvider.aksCurrentlyLocked = false;
    [self.lockStateTracker recheck];

    [self waitForExpectations:@[expectation] timeout:5];

}


@end

#endif /* OCTAGON */
