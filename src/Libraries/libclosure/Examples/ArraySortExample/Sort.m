/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#import <Foundation/Foundation.h>

#ifndef __BLOCKS__
#error compiler does not support blocks.
#endif

#if !NS_BLOCKS_AVAILABLE
#error Blocks do not appear to be available, according to the Foundation.
#endif

NSInteger sortStuff(id a, id  b, void *inReverse) {
    int reverse = (int) inReverse; // oops
    int result = [(NSString *)a compare: b];
    return reverse ? -result : result;
}

int main (int argc, const char * argv[]) {
    NSArray *stuff = [NSArray arrayWithObjects: @"SQUARED OFF", @"EIGHT CORNERS", @"90-DEGREE ANGLES", @"FLAT TOP", @"STARES STRAIGHT AHEAD", @"STOCK PARTS", nil];
    int inReverse = 1;
    
    NSLog(@"reverse func: %@", [stuff sortedArrayUsingFunction:sortStuff context: &inReverse]);
    NSLog(@"reverse block: %@", [stuff sortedArrayUsingComparator: ^(id a,  id b) {
        int result = [a compare: b];
        return inReverse ? -result : result;
    }]);

    inReverse = 0;

    NSLog(@"forward func: %@", [stuff sortedArrayUsingFunction:sortStuff context: &inReverse]);
    NSLog(@"forward block: %@", [stuff sortedArrayUsingComparator: ^(id a,  id b) {
        int result = [a compare: b];
        return inReverse ? -result : result;
    }]);
    
    return 0;
}
