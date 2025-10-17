/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#import <XCTest/XCTest.h>
#import <OCMock/OCMock.h>
#import <CloudKit/CloudKit.h>
#import <CloudKit/CloudKit_Private.h>
#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/OctagonAPSReceiver.h"
#import "keychain/ckks/tests/OctagonAPSReceiverTests.h"
#import "keychain/ckks/tests/MockCloudKit.h"
#import "keychain/ckks/CKKSCondition.h"

#if OCTAGON

@interface CKKSAPSNotificationReceiver : NSObject <CKKSZoneUpdateReceiverProtocol>
@property XCTestExpectation* expectation;
@property void (^block)(CKRecordZoneNotification* notification);

- (instancetype)initWithExpectation:(XCTestExpectation*)expectation;
- (instancetype)initWithExpectation:(XCTestExpectation*)expectation block:(void (^)(CKRecordZoneNotification* notification))block;
@end

@implementation CKKSAPSNotificationReceiver
- (instancetype)initWithExpectation:(XCTestExpectation*)expectation {
    return [self initWithExpectation:expectation block:nil];
}

- (instancetype)initWithExpectation:(XCTestExpectation*)expectation block:(void (^)(CKRecordZoneNotification* notification))block
{
    if((self = [super init])) {
        _expectation = expectation;
        _block = block;
    }
    return self;
}

- (void)notifyZoneChange: (CKRecordZoneNotification*) notification {
    [self.expectation fulfill];
    self.block(notification);
}
@end


@implementation OctagonAPSReceiverTests

+ (APSIncomingMessage*)messageForZoneID:(CKRecordZoneID*)zoneID {
    // reverse engineered from source code. Ugly.

    NSMutableDictionary* zoneInfo = [[NSMutableDictionary alloc] init];
    zoneInfo[@"zid"] = zoneID.zoneName; // kCKNotificationCKRecordZoneRecordZoneIDKey
    zoneInfo[@"zoid"] = CKCurrentUserDefaultName; // kCKNotificationCKRecordZoneRecordZoneOwnerIDKey

    NSMutableDictionary* ckinfo = [[NSMutableDictionary alloc] init];
    ckinfo[@"fet"] = zoneInfo; // kCKNotificationCKRecordZoneKey

    NSMutableDictionary* d = [[NSMutableDictionary alloc] init];
    d[@"ck"] = ckinfo; // kCKNotificationCKKey

    return [[APSIncomingMessage alloc] initWithTopic:@"i'm-not-sure" userInfo:d];
}

- (void)setUp {
    [super setUp];

    self.testZoneID = [[CKRecordZoneID alloc] initWithZoneName:@"testzone" ownerName:CKCurrentUserDefaultName];

    // Make sure our helpers work properly
    APSIncomingMessage* message = [OctagonAPSReceiverTests messageForZoneID:self.testZoneID];
    XCTAssertNotNil(message, "Should have received a APSIncomingMessage");

    CKNotification* notification = [CKNotification notificationFromRemoteNotificationDictionary:message.userInfo];
    XCTAssertNotNil(notification, "Should have received a CKNotification");
    XCTAssert([notification isKindOfClass: [CKRecordZoneNotification class]], "Should have received a CKRecordZoneNotification");
    CKRecordZoneNotification* ckrzn = (CKRecordZoneNotification*) notification;

    XCTAssertEqual(self.testZoneID, ckrzn.recordZoneID, "Should have received a notification for the test zone");
}

- (void)tearDown {
    self.testZoneID = nil;

    [super tearDown];
}


- (void)testRegisterAndReceive {
    __weak __typeof(self)weakSelf = self;

    OctagonAPSReceiver* apsr = [[OctagonAPSReceiver alloc] initWithNamedDelegatePort:SecCKKSAPSNamedPort
                                                                  apsConnectionClass:[FakeAPSConnection class]];

    XCTAssertNotNil(apsr, "Should have received a OctagonAPSReceiver");

    CKKSAPSNotificationReceiver* anr = [[CKKSAPSNotificationReceiver alloc] initWithExpectation:[self expectationWithDescription:@"receive notification"]
                                                                                          block:
                                        ^(CKRecordZoneNotification* notification) {
                                            __strong __typeof(self) strongSelf = weakSelf;
                                            XCTAssertNotNil(notification, "Should have received a notification");
                                            XCTAssertEqual(strongSelf.testZoneID, notification.recordZoneID, "Should have received a notification for the test zone");
                                        }];

    CKKSCondition* registered = [apsr registerCKKSReceiver:anr contextID:@"contextID"];
    XCTAssertEqual(0, [registered wait:1*NSEC_PER_SEC], "Registration should have completed within a second");
    APSIncomingMessage* message = [OctagonAPSReceiverTests messageForZoneID:self.testZoneID];
    XCTAssertNotNil(message, "Should have received a APSIncomingMessage");

    [apsr connection:apsr.apsConnection didReceiveIncomingMessage:message];

    [self waitForExpectationsWithTimeout:5.0 handler:nil];
}

- (void)testReceiveNotificationIfRegisteredAfterDelivery {
    OctagonAPSReceiver* apsr = [[OctagonAPSReceiver alloc] initWithNamedDelegatePort:SecCKKSAPSNamedPort
                                                                  apsConnectionClass:[FakeAPSConnection class]];
    XCTAssertNotNil(apsr, "Should have received a OctagonAPSReceiver");

    // Receives a notification for the test zone
    APSIncomingMessage* message = [OctagonAPSReceiverTests messageForZoneID:self.testZoneID];
    XCTAssertNotNil(message, "Should have received a APSIncomingMessage");
    [apsr connection:apsr.apsConnection didReceiveIncomingMessage:message];

    CKKSAPSNotificationReceiver* anr = [[CKKSAPSNotificationReceiver alloc] initWithExpectation:[self expectationWithDescription:@"receive notification"]
                                                                                          block:
                                        ^(CKRecordZoneNotification* notification) {
                                            XCTAssertNotNil(notification, "Should have received a (stored) notification");
                                        }];

    CKKSCondition* registered = [apsr registerCKKSReceiver:anr contextID:@"contextID"];
    XCTAssertEqual(0, [registered wait:1*NSEC_PER_SEC], "Registration should have completed within a second");

    [self waitForExpectationsWithTimeout:5.0 handler:nil];
}

@end

#endif
