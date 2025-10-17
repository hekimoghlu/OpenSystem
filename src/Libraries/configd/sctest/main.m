/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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

static void __attribute__((noreturn))
usage(void)
{
	NSArray *testClasses = getTestClasses();
	SCTestLog("\nUsage: sctest <command> <options>");
	SCTestLog("\nCommands:");
	for (NSString *testClassName in testClasses) {
		Class testClass = NSClassFromString(testClassName);
		SCTestLog("  %15s:   %s", [testClass command].UTF8String, [testClass commandDescription].UTF8String);
	}

	SCTestLog("\n\nOptions:");
#if !TARGET_OS_BRIDGE
	SCTestLog(kSCTestOptionHelpAllPlatforms kSCTestOptionHelpNonBridgeOS "\n");
#else // !TARGET_OS_BRIDGE
	SCTestLog(kSCTestOptionHelpAllPlatforms "\n");
#endif // !TARGET_OS_BRIDGE

	ERR_EXIT;
}

int main(int argc, const char * argv[]) {
	@autoreleasepool {
		NSString *testCommand;
		NSArray<NSString *> *testClasses;
		BOOL commandValid = NO;
		NSDictionary *options;
		Class testClass;
		SCTest *testClassObject;

		_sc_log = kSCLogDestinationFile;	// print (stdout)

		if (argc == 1) {
			usage();
		}

		testCommand = @(argv[1]);
		// Check if the command is valid
		testClasses = getTestClasses();
		for (NSString *testClassName in testClasses) {
			Class testClass = NSClassFromString(testClassName);
			if ([[testClass command] isEqualToString:testCommand]) {
				commandValid = YES;
				break;
			}

		}

		if (!commandValid) {
			SCTestLog("Invalid command: %@", testCommand);
			usage();
		}

		// Create the options dictionary
		options = getOptionsDictionary(argc, argv);
		if (options == nil) {
			usage();
		}

		// Initialize the command
		for (NSString *className in testClasses) {
			Class	commandClass = NSClassFromString(className);
			if ([testCommand isEqualToString:[commandClass command]]) {
				testClass = commandClass;
				break;
			}
		}

		_sc_log = kSCLogDestinationBoth_NoTime;	// log AND print (stdout w/o timestamp)

		testClassObject = [(SCTest *)[testClass alloc] initWithOptions:options];
		if (testClassObject.options[kSCTestGlobalOptionCPU] != nil) {
			cpuStart(testClassObject.globalCPU);
		}

		if (testClassObject.options[kSCTestGlobalOptionTime] != nil) {
			timerStart(testClassObject.globalTimer);
		}

		[testClassObject start];

		dispatch_main();
	}
}
