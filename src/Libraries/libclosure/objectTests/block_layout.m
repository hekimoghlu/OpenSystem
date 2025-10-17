/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
/*  block_layout.m
    Created by Patrick Beard on 3 Sep 2010
*/

// TEST_CFLAGS -framework Foundation

#import <Foundation/Foundation.h>
#import <Block.h>
#import <Block_private.h>
#import <dispatch/dispatch.h>
#import <assert.h>
#import "test.h"

int main (int argc, char const* argv[]) {
    NSAutoreleasePool *pool = [NSAutoreleasePool new];
    
    NSObject *o = [NSObject new];
    NSString *s = [NSString stringWithFormat:@"argc = %d, argv = %p", argc, argv];

    dispatch_block_t block = ^{
        NSLog(@"o = %@", o);
        NSLog(@"s = %@", s);
    };

        
    const char *layout = _Block_extended_layout(block);
    testprintf("layout %p\n", layout);
    assert (layout == (void*)0x200);

    const char *gclayout = _Block_layout(block);
    testprintf("GC layout %p\n", gclayout);
    assert (gclayout == NULL);

    block = [block copy];
    
    layout = _Block_extended_layout(block);
    testprintf("layout %p\n", layout);
    assert (layout == (void*)0x200);

    gclayout = _Block_layout(block);
    testprintf("GC layout %p\n", gclayout);
    assert (gclayout == NULL);
    
    block();
    [block release];
    
    [pool drain];
    
    succeed(__FILE__);
}
