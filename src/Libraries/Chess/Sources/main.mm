/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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
#import <Cocoa/Cocoa.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include "MBCDebug.h"

int main(int argc, const char *argv[])
{
	while (argv[1])
		putenv((char *)*++argv);
    MBCDebug::Update();
	if (MBCDebug::LogStart())
		NSLog(@"Chess starting\n");
	//
	// We set defaults that influence NSApplication init, so we need to run now
	//
	NSAutoreleasePool * autoreleasePool = [[NSAutoreleasePool alloc] init];
	NSDictionary * defaults = 
	[NSDictionary dictionaryWithContentsOfFile:
	 [[NSBundle mainBundle] 
	  pathForResource:@"Defaults" ofType:@"plist"]];
	[[NSUserDefaults standardUserDefaults] registerDefaults: defaults];
	[[NSUserDefaults standardUserDefaults] synchronize];
	[autoreleasePool drain];
    return NSApplicationMain(argc, argv);
}
