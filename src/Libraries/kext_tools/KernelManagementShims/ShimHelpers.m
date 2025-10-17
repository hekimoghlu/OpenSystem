/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
//  ShimHelpers.m
//  kextcache
//
//  Created by jkb on 3/11/20.
//

#import <Foundation/Foundation.h>

#import "Shims.h"
#import "../kext_tools_util.h"

#if !(TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR) && !TARGET_OS_BRIDGE

static NSTask *shimmedTask = nil;
static NSMutableArray *shimmedTaskArguments = nil;

void initArgShimming()
{
    shimmedTask = [[NSTask alloc] init];
    [shimmedTask setLaunchPath:@"/usr/bin/kmutil"];
    shimmedTaskArguments = [[NSMutableArray alloc] init];
}

void addArguments(NSArray<NSString *> *arguments)
{
    [shimmedTaskArguments addObjectsFromArray:arguments];
}

void addArgument(NSString *argument)
{
    [shimmedTaskArguments addObject:argument];
}

NSString *createStringFromShimmedArguments()
{
    NSString *allArguments = [shimmedTaskArguments componentsJoinedByString:@" "];
    return [NSString stringWithFormat:@"%@ %@", shimmedTask.launchPath, allArguments];
}

void runWithShimmedArguments()
{
    OSKextLogCFString(/* kext */ NULL,
	    kOSKextLogWarningLevel | kOSKextLogGeneralFlag,
	    CFSTR("Executing: %@"),
		(__bridge CFStringRef)createStringFromShimmedArguments());

    [shimmedTask setArguments:shimmedTaskArguments];
    [shimmedTask launch];
    [shimmedTask waitUntilExit];
    exit([shimmedTask terminationStatus]);
}

#else // #if !(TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR) && !TARGET_OS_BRIDGE

void initArgShimming()
{
}

void addArguments(NSArray<NSString *> *arguments)
{
	(void)arguments;
}

void addArgument(NSString *argument)
{
	(void)argument;
}

NSString *createStringFromShimmedArguments()
{
	return NULL;
}

void runWithShimmedArguments()
{
}

#endif // #if !(TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR) && !TARGET_OS_BRIDGE
