/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
#import "SCTest.h"
#import "SCTestUtils.h"

@interface SCTestUnitTest : SCTest
@end

@implementation SCTestUnitTest

+ (NSString *)command
{
	return @"unit_test";
}

+ (NSString *)commandDescription
{
	return @"Runs the unit test for all commands";
}

- (void)listTests
{
	NSMutableDictionary *testDictionary;
	NSMutableArray *testsArray;
	NSString *thisClass;
	NSArray<NSString *> *testClasses;
	NSData *data;
	NSString *jsonString;

	testDictionary = [[NSMutableDictionary alloc] init];
	[testDictionary setObject:@"SystemConfiguration Unit Tests" forKey:@"Name"];
	[testDictionary setObject:@"These tests exercise 'configd' and the 'SystemConfiguration' framework" forKey:@"Description"];

	testsArray = [[NSMutableArray alloc] init];
	thisClass = NSStringFromClass([self class]);
	testClasses = getTestClasses();
	for (NSString *className in testClasses) {
		Class testClass;
		NSMutableDictionary *subTest;
		NSArray *list;
		NSMutableArray *subTestArray;

		if ([className isEqualToString:thisClass] ||
		    [className isEqualToString:@"SCTestRankAssertion"]) {
			continue;
		}

		testClass = NSClassFromString(className);
		list = getUnitTestListForClass(testClass);

		subTest = [[NSMutableDictionary alloc] init];
		[subTest setObject:@NO forKey:@"RequiresTCPDUMP"];
		[subTest setObject:@YES forKey:@"RequiresNetwork"];
		[subTest setObject:@NO forKey:@"RequiresRoot"];
		[subTest setObject:@NO forKey:@"RequiresPowermetrics"];
		[subTest setObject:[testClass command] forKey:@"Name"];
		[subTest setObject:[testClass commandDescription] forKey:@"Description"];
		subTestArray = [[NSMutableArray alloc] init];
		for (NSString *unitTest in list) {
			NSDictionary *testDict = @{@"Command":@[@"/usr/local/bin/sctest",
								@"unit_test",
								@"-test_method",
								unitTest],
							@"Name":[unitTest stringByReplacingOccurrencesOfString:@"unitTest" withString:@""],
							@"Description":@"Unit test"
							};
			[subTestArray addObject:testDict];
		}
		[subTest setObject:subTestArray forKey:@"SubTests"];
		[testsArray addObject:subTest];
	}
	[testDictionary setObject:testsArray forKey:@"Tests"];
	data = [NSJSONSerialization dataWithJSONObject:testDictionary
						options:NSJSONWritingPrettyPrinted
						error:nil];
	jsonString = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
	SCPrint(TRUE, stdout, CFSTR("%@"), jsonString);
}

- (void)start
{
	NSArray<NSString *> *testClasses;
	BOOL errorOccured = NO;
    
	NSString *thisClass = NSStringFromClass([self class]);;
	testClasses = getTestClasses();

	if (self.options[kSCTestUnitTestListTests]) {
		[self listTests];
	} else if (self.options[kSCTestUnitTestTestMethodList]) {
		SCTestLog("List of unit tests:");
		for (NSString *className in testClasses) {
			Class testClass;
			NSArray *list;

			if ([className isEqualToString:thisClass]) {
				continue;
			}

			testClass = NSClassFromString(className);
			if (self.options[kSCTestUnitTestCommand] != nil) {
				if (![self.options[kSCTestUnitTestCommand] isEqualToString:[testClass command]]) {
					// Run unit test only for a specific command
					continue;
				}
			}

			SCTestLog("\n======= '%@' unit tests =======", [testClass command]);
			list = getUnitTestListForClass(testClass);
			for (NSString *unitTest in list) {
				SCTestLog("%@", unitTest);
			}
		}

		SCTestLog("\nEach of the unit tests can be run with the 'test_method' option\n");
	} else if (self.options[kSCTestUnitTestTestMethod]) {
		for (NSString *className in testClasses) {
			Class testClass;
			NSArray *list;

			if ([className isEqualToString:thisClass]) {
				continue;
			}

			testClass = NSClassFromString(className);
			if (self.options[kSCTestUnitTestCommand] != nil) {
				if (![self.options[kSCTestUnitTestCommand] isEqualToString:[testClass command]]) {
					// Run unit test only for a specific command
					continue;
				}
			}

			list = getUnitTestListForClass(testClass);
			for (NSString *unitTest in list) {
				if ([unitTest isEqualToString:self.options[kSCTestUnitTestTestMethod]]) {
                    SEL methodSelector;
                    Boolean retVal;
                    
					id obj = [(SCTest *)[testClass alloc] initWithOptions:self.options];
					SCTestLog("Running unit test %@ ...", unitTest);

					methodSelector = NSSelectorFromString(unitTest);
					retVal = false;
					if ([obj respondsToSelector:methodSelector]) {
						NSInvocation *invocation = [NSInvocation invocationWithMethodSignature:[obj methodSignatureForSelector:methodSelector]];
						invocation.target = obj;
						invocation.selector = methodSelector;
						[invocation invoke];
						[invocation getReturnValue:&retVal];
					}

					if (!retVal) {
						SCTestLog("FAILED");
						errorOccured = YES;
					} else {
						SCTestLog("PASSED");
					}
					break;
				}
			}
		}
	} else {
		// This command runs unit tests for all commands.
		for (NSString *className in testClasses) {
			Class testClass;
			id obj;

			if ([className isEqualToString:thisClass]) {
				continue;
			}

			testClass = NSClassFromString(className);
			if (self.options[kSCTestUnitTestCommand] != nil) {
				if (![self.options[kSCTestUnitTestCommand] isEqualToString:[testClass command]]) {
					// Run unit test only for a specific command
					continue;
				}
			}

			obj = [(SCTest *)[testClass alloc] initWithOptions:self.options];
            if ([obj respondsToSelector:@selector(unitTest)]) {
                BOOL passed;
                
				SCTestLog("\n*** Running unit test for \"%@\" command ***\n", [testClass command]);
				passed = [obj unitTest];
				if (!passed) {
					SCTestLog("FAILED");
					errorOccured = YES;
				}
			}
		}
	}

	[self cleanupAndExitWithErrorCode:errorOccured];
}

- (void)cleanupAndExitWithErrorCode:(int)error
{
	[super cleanupAndExitWithErrorCode:error];
}

@end
