/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
//  kextload.m
//  kextload
//
//  Created by jkb on 3/11/20.
//

#import <Foundation/Foundation.h>
#import "../kextload_main.h"
#import "ShimHelpers.h"


// This function will always result in a call to exit(). If calling this
// function spawns kmutil, then it will call exit() with kmutil's exit status.
void shimKextloadArgsToKMUtilAndRun(KextloadArgs *toolArgs)
{
    int exitCode = EX_OK; // what we'll exit with if we don't spawn kmutil

    initArgShimming();
    addArgument(@"load");

    if (toolArgs->kextIDs) {
	for (NSString *kextID in (__bridge NSArray<NSString *> *)toolArgs->kextIDs) {
	    addArguments(@[@"-b", kextID]);
	}
    }
    if (toolArgs->kextURLs) {
	for (NSURL *kextURL in (__bridge NSArray<NSURL *> *)toolArgs->kextURLs) {
	    addArguments(@[@"-p", kextURL.path]);
	}
    }

    /* TODO: Proper support for explicit dependencies of kexts */
    if (toolArgs->dependencyURLs) {
	for (NSURL *dependencyURL in (__bridge NSArray<NSURL *> *)toolArgs->dependencyURLs) {
	    addArguments(@[@"-p", dependencyURL.path]);
	}
    }
    if (toolArgs->repositoryURLs) {
	for (NSURL *repositoryURL in (__bridge NSArray<NSURL *> *)toolArgs->repositoryURLs) {
	    addArguments(@[@"-r", repositoryURL.path]);
	}
    }

    runWithShimmedArguments();
cancel:
    exit(exitCode);
}
