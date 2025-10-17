/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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

@implementation SCTest

- (instancetype)initWithOptions:(NSDictionary *)options
{
	self = [super init];
	if (self) {
		self.options = options;
		self.globalCPU = malloc(sizeof(CPUUsageInfo));
		self.globalTimer = malloc(sizeof(timerInfo));
	}
	return self;
}

- (void)dealloc
{
	if (self.globalTimer != NULL) {
		free(self.globalTimer);
	}
	if (self.globalCPU != NULL) {
		free(self.globalCPU);
	}
}

+ (NSString *)command
{
	return @"sctest";
}

+ (NSString *)commandDescription
{
	return @"This is a generic class";
}

+ (NSString *)commandHelp
{
	return @"This is a generic help";
}

- (void)start
{
	return;
}

- (void)waitFor:(double)seconds
{
	dispatch_semaphore_t sem = dispatch_semaphore_create(0);
	dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
	dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, seconds * NSEC_PER_SEC), DISPATCH_TIME_FOREVER, 0);
	dispatch_source_set_event_handler(timer, ^{
		dispatch_semaphore_signal(sem);
	});
	dispatch_resume(timer);
	dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
}

- (void)cleanupAndExitWithErrorCode:(int)error
{
	if (self.options[kSCTestGlobalOptionTime] != nil) {
		timerEnd(self.globalTimer);
		SCTestLog("Time: %@ s", createUsageStringForTimer(self.globalTimer));
	}

	if (self.options[kSCTestGlobalOptionCPU] != nil) {
		cpuEnd(self.globalCPU);
		SCTestLog("CPU: %@", createUsageStringForCPU(self.globalCPU));
	}

	if (self.options[kSCTestGlobalOptionWait] != nil) {
		return;
	}
	exit(error);
}

- (BOOL)unitTest
{
	return YES;
}

@end
