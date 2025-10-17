/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ios_main.mm: Alternative entry point for iOS executables that initializes UIKit before calling
// the default entry point.

#import <UIKit/UIKit.h>

#include <stdio.h>

static int original_argc;
static char **original_argv;

int main(int argc, char **argv);

@interface AngleUtilAppDelegate : UIResponder <UIApplicationDelegate>

@property(nullable, nonatomic, strong) UIWindow *window;

@end

@implementation AngleUtilAppDelegate

@synthesize window;

- (void)runMain
{
    exit(main(original_argc, original_argv));
}

- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
    self.window                    = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
    self.window.rootViewController = [[UIViewController alloc] initWithNibName:nil bundle:nil];
    [self.window makeKeyAndVisible];
    // We need to return from this function before the app finishes launching, so call main in a
    // timer callback afterward.
    [NSTimer scheduledTimerWithTimeInterval:0
                                     target:self
                                   selector:@selector(runMain)
                                   userInfo:nil
                                    repeats:NO];
    return YES;
}

@end

extern "C" int ios_main(int argc, char **argv)
{
    original_argc = argc;
    original_argv = argv;
    return UIApplicationMain(argc, argv, nullptr, NSStringFromClass([AngleUtilAppDelegate class]));
}
