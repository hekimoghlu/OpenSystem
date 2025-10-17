/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
#ifndef SCTestUtils_h
#define SCTestUtils_h

#import <Foundation/Foundation.h>
#import <SystemConfiguration/SCPrivate.h>
#import <objc/objc-runtime.h>

#define SCTestLog(fmt, ...)	SCLog(TRUE, LOG_NOTICE, CFSTR(fmt), ##__VA_ARGS__)

#define ERR_EXIT 		exit(1)

typedef struct {
	uint64_t user;
	uint64_t sys;
	uint64_t idle;
} CPUUsageInfoInner;

typedef struct {
	CPUUsageInfoInner startCPU;
	CPUUsageInfoInner endCPU;
} CPUUsageInfo;

typedef struct {
	struct timespec startTime;
	struct timespec endTime;
} timerInfo;

void timerStart(timerInfo *);
void timerEnd(timerInfo *);
NSString * createUsageStringForTimer(timerInfo *);

void cpuStart(CPUUsageInfo *);
void cpuEnd(CPUUsageInfo *);
NSString * createUsageStringForCPU(CPUUsageInfo *cpu);

NSArray<NSString *> *getTestClasses(void);
NSArray<NSString *> *getUnitTestListForClass(Class base);
NSDictionary *getOptionsDictionary(int argc, const char * const argv[]);

#endif /* SCTestUtils_h */
