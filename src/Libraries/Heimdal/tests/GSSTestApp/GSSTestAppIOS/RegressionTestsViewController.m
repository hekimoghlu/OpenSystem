/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
//  RegressionTestsViewController.m
//  GSSTestApp
//
//  Created by Love HÃ¶rnquist Ã…strand on 2014-08-15.
//  Copyright (c) 2014 Apple, Inc. All rights reserved.
//

#import "RegressionTestsViewController.h"
#import "FakeXCTest.h"

@interface RegressionTestsViewController ()
@property (strong) dispatch_queue_t queue;

@end

static RegressionTestsViewController *me = NULL;

__attribute__((format(printf, 1, 0)))
static int
callback(const char *fmt, va_list ap)
{
    if (me == NULL)
        return -1;

    char *output = NULL;

    vasprintf(&output, fmt, ap);

    dispatch_async(dispatch_get_main_queue(), ^{
        [me appendProgress:[NSString stringWithUTF8String:output] color:NULL];
        free(output);
    });

    return 0;
}

@implementation RegressionTestsViewController

- (void)viewDidLoad
{
    [super viewDidLoad];

    me = self;
    XFakeXCTestCallback = callback;

    self.queue = dispatch_queue_create("test-queue", NULL);

    [self runTests:self];
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations
{
    return UIInterfaceOrientationMaskPortrait;
}

- (IBAction)runTests:(id)sender
{
    [self.statusLabel setText:@"running"];
    [self.progressTextView setText:@""];

    dispatch_async(self.queue, ^{
        NSString *testStatus;
        int result = [XCTest runTests];

        if (result == 0)
            testStatus = @"all tests passed";
        else
            testStatus = [NSString stringWithFormat:@"%d tests failed", result];

        dispatch_async(dispatch_get_main_queue(), ^{
            [self.statusLabel setText:testStatus];
        });
    });
}

- (void)appendProgress:(NSString *)string color:(UIColor *)color {

    NSMutableAttributedString* str = [[NSMutableAttributedString alloc] initWithString:string];
    if (color)
        [str addAttribute:NSForegroundColorAttributeName value:color range:NSMakeRange(0, [str length])];

    NSTextStorage *textStorage = [self.progressTextView textStorage];

    [textStorage beginEditing];
    [textStorage appendAttributedString:str];
    [textStorage endEditing];
}

@end
