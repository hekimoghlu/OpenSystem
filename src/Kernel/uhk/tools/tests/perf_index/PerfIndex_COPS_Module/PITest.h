/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
//  PITest.h
//  PerfIndex
//
//  Created by Mark Hamilton on 8/21/13.
//
//

#import <Foundation/Foundation.h>
#import "PerfIndex.h"

@interface PITest : NSObject <HGTest>
{
	int (*setup_func)(int, long long, int, void**);
	int (*execute_func)(int, int, long long, int, void**);
	void (*cleanup_func)(int, long long);

	long long length;
	int numThreads;
	int readyThreadCount;
	int testArgc;
	void** testArgv;
	pthread_mutex_t readyThreadCountLock;
	pthread_cond_t threadsReadyCvar;
	pthread_cond_t startCvar;
	pthread_t* threads;
}

@property NSString* testName;

- (BOOL)setup;
- (BOOL)execute;
- (void)cleanup;


@end
