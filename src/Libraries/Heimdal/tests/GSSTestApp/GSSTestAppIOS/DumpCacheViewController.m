/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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
//  DumpCacheViewController.m
//  GSSTestApp
//
//  Created by Love HÃ¶rnquist Ã…strand on 2014-09-03.
//  Copyright (c) 2014 Apple, Inc. All rights reserved.
//

#import "DumpCacheViewController.h"
#include <Heimdal/heimcred.h>

@interface DumpCacheViewController ()

@end

@implementation DumpCacheViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    [self dumpCredentials:(id)self];
}

- (IBAction)dumpCredentials:(id)sender {

    CFDictionaryRef status = HeimCredCopyStatus(NULL);
    if (status) {
        CFDataRef data = CFPropertyListCreateData(NULL, status, kCFPropertyListXMLFormat_v1_0, 0, NULL);
        CFRelease(status);
        if (data == NULL) {
            [self.dumpCacheTextView setText:@"failed to convert dictionary to a plist"];
        }
        NSString *string = [[NSString alloc] initWithData:(__bridge NSData *)data encoding:NSUTF8StringEncoding];

        [self.dumpCacheTextView setText:string];
        CFRelease(data);
    } else {
        [self.dumpCacheTextView setText:@"no credentials to dump\n"];
    }
}

@end
