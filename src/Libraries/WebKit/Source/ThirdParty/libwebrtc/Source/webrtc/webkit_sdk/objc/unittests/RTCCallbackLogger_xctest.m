/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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
#import "api/logging/RTCCallbackLogger.h"

#import <XCTest/XCTest.h>

@interface RTCCallbackLoggerTests : XCTestCase

@property(nonatomic, strong) RTCCallbackLogger *logger;

@end

@implementation RTCCallbackLoggerTests

@synthesize logger;

- (void)setUp {
  self.logger = [[RTCCallbackLogger alloc] init];
}

- (void)tearDown {
  self.logger = nil;
}

- (void)testDefaultSeverityLevel {
  XCTAssertEqual(self.logger.severity, RTCLoggingSeverityInfo);
}

- (void)testCallbackGetsCalledForAppropriateLevel {
  self.logger.severity = RTCLoggingSeverityWarning;

  XCTestExpectation *callbackExpectation = [self expectationWithDescription:@"callbackWarning"];

  [self.logger start:^(NSString *message) {
    XCTAssertTrue([message hasSuffix:@"Horrible error\n"]);
    [callbackExpectation fulfill];
  }];

  RTCLogError("Horrible error");

  [self waitForExpectations:@[ callbackExpectation ] timeout:10.0];
}

- (void)testCallbackWithSeverityGetsCalledForAppropriateLevel {
  self.logger.severity = RTCLoggingSeverityWarning;

  XCTestExpectation *callbackExpectation = [self expectationWithDescription:@"callbackWarning"];

  [self.logger
      startWithMessageAndSeverityHandler:^(NSString *message, RTCLoggingSeverity severity) {
        XCTAssertTrue([message hasSuffix:@"Horrible error\n"]);
        XCTAssertEqual(severity, RTCLoggingSeverityError);
        [callbackExpectation fulfill];
      }];

  RTCLogError("Horrible error");

  [self waitForExpectations:@[ callbackExpectation ] timeout:10.0];
}

- (void)testCallbackDoesNotGetCalledForOtherLevels {
  self.logger.severity = RTCLoggingSeverityError;

  XCTestExpectation *callbackExpectation = [self expectationWithDescription:@"callbackError"];

  [self.logger start:^(NSString *message) {
    XCTAssertTrue([message hasSuffix:@"Horrible error\n"]);
    [callbackExpectation fulfill];
  }];

  RTCLogInfo("Just some info");
  RTCLogWarning("Warning warning");
  RTCLogError("Horrible error");

  [self waitForExpectations:@[ callbackExpectation ] timeout:10.0];
}

- (void)testCallbackWithSeverityDoesNotGetCalledForOtherLevels {
  self.logger.severity = RTCLoggingSeverityError;

  XCTestExpectation *callbackExpectation = [self expectationWithDescription:@"callbackError"];

  [self.logger
      startWithMessageAndSeverityHandler:^(NSString *message, RTCLoggingSeverity severity) {
        XCTAssertTrue([message hasSuffix:@"Horrible error\n"]);
        XCTAssertEqual(severity, RTCLoggingSeverityError);
        [callbackExpectation fulfill];
      }];

  RTCLogInfo("Just some info");
  RTCLogWarning("Warning warning");
  RTCLogError("Horrible error");

  [self waitForExpectations:@[ callbackExpectation ] timeout:10.0];
}

- (void)testCallbackDoesNotgetCalledForSeverityNone {
  self.logger.severity = RTCLoggingSeverityNone;

  XCTestExpectation *callbackExpectation = [self expectationWithDescription:@"unexpectedCallback"];

  [self.logger start:^(NSString *message) {
    [callbackExpectation fulfill];
    XCTAssertTrue(false);
  }];

  RTCLogInfo("Just some info");
  RTCLogWarning("Warning warning");
  RTCLogError("Horrible error");

  XCTWaiter *waiter = [[XCTWaiter alloc] init];
  XCTWaiterResult result = [waiter waitForExpectations:@[ callbackExpectation ] timeout:1.0];
  XCTAssertEqual(result, XCTWaiterResultTimedOut);
}

- (void)testCallbackWithSeverityDoesNotgetCalledForSeverityNone {
  self.logger.severity = RTCLoggingSeverityNone;

  XCTestExpectation *callbackExpectation = [self expectationWithDescription:@"unexpectedCallback"];

  [self.logger
      startWithMessageAndSeverityHandler:^(NSString *message, RTCLoggingSeverity severity) {
        [callbackExpectation fulfill];
        XCTAssertTrue(false);
      }];

  RTCLogInfo("Just some info");
  RTCLogWarning("Warning warning");
  RTCLogError("Horrible error");

  XCTWaiter *waiter = [[XCTWaiter alloc] init];
  XCTWaiterResult result = [waiter waitForExpectations:@[ callbackExpectation ] timeout:1.0];
  XCTAssertEqual(result, XCTWaiterResultTimedOut);
}

- (void)testStartingWithNilCallbackDoesNotCrash {
  [self.logger start:nil];

  RTCLogError("Horrible error");
}

- (void)testStartingWithNilCallbackWithSeverityDoesNotCrash {
  [self.logger startWithMessageAndSeverityHandler:nil];

  RTCLogError("Horrible error");
}

- (void)testStopCallbackLogger {
  XCTestExpectation *callbackExpectation = [self expectationWithDescription:@"stopped"];

  [self.logger start:^(NSString *message) {
    [callbackExpectation fulfill];
  }];

  [self.logger stop];

  RTCLogInfo("Just some info");

  XCTWaiter *waiter = [[XCTWaiter alloc] init];
  XCTWaiterResult result = [waiter waitForExpectations:@[ callbackExpectation ] timeout:1.0];
  XCTAssertEqual(result, XCTWaiterResultTimedOut);
}

- (void)testStopCallbackWithSeverityLogger {
  XCTestExpectation *callbackExpectation = [self expectationWithDescription:@"stopped"];

  [self.logger
      startWithMessageAndSeverityHandler:^(NSString *message, RTCLoggingSeverity loggingServerity) {
        [callbackExpectation fulfill];
      }];

  [self.logger stop];

  RTCLogInfo("Just some info");

  XCTWaiter *waiter = [[XCTWaiter alloc] init];
  XCTWaiterResult result = [waiter waitForExpectations:@[ callbackExpectation ] timeout:1.0];
  XCTAssertEqual(result, XCTWaiterResultTimedOut);
}

- (void)testDestroyingCallbackLogger {
  XCTestExpectation *callbackExpectation = [self expectationWithDescription:@"destroyed"];

  [self.logger start:^(NSString *message) {
    [callbackExpectation fulfill];
  }];

  self.logger = nil;

  RTCLogInfo("Just some info");

  XCTWaiter *waiter = [[XCTWaiter alloc] init];
  XCTWaiterResult result = [waiter waitForExpectations:@[ callbackExpectation ] timeout:1.0];
  XCTAssertEqual(result, XCTWaiterResultTimedOut);
}

- (void)testDestroyingCallbackWithSeverityLogger {
  XCTestExpectation *callbackExpectation = [self expectationWithDescription:@"destroyed"];

  [self.logger
      startWithMessageAndSeverityHandler:^(NSString *message, RTCLoggingSeverity loggingServerity) {
        [callbackExpectation fulfill];
      }];

  self.logger = nil;

  RTCLogInfo("Just some info");

  XCTWaiter *waiter = [[XCTWaiter alloc] init];
  XCTWaiterResult result = [waiter waitForExpectations:@[ callbackExpectation ] timeout:1.0];
  XCTAssertEqual(result, XCTWaiterResultTimedOut);
}

- (void)testCallbackWithSeverityLoggerCannotStartTwice {
  self.logger.severity = RTCLoggingSeverityWarning;

  XCTestExpectation *callbackExpectation = [self expectationWithDescription:@"callbackWarning"];

  [self.logger
      startWithMessageAndSeverityHandler:^(NSString *message, RTCLoggingSeverity loggingServerity) {
        XCTAssertTrue([message hasSuffix:@"Horrible error\n"]);
        XCTAssertEqual(loggingServerity, RTCLoggingSeverityError);
        [callbackExpectation fulfill];
      }];

  [self.logger start:^(NSString *message) {
    [callbackExpectation fulfill];
    XCTAssertTrue(false);
  }];

  RTCLogError("Horrible error");

  [self waitForExpectations:@[ callbackExpectation ] timeout:10.0];
}

@end
