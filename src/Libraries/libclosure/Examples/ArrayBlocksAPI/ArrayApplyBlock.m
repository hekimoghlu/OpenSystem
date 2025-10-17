/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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

int main (int argc, const char * argv[]) {
    NSArray *array = [NSArray arrayWithObjects: @"A", @"B", @"C", @"A", @"B", @"Z",@"G", @"are", @"Q", nil];
	NSSet *filterSet = [NSSet setWithObjects: @"A", @"Z", @"Q", nil];
    
    [array enumerateObjectsUsingBlock:  ^(id anObject, NSUInteger idx, BOOL *stop) {
        NSLog(@"%d: \t %@", idx, anObject);
        if (idx == 4) {
            NSLog(@"\tStopping Enumeration.");
            *stop = YES;
        }
    }];
    
    NSIndexSet *indexSet = [array indexesOfObjectsPassingTest: ^(id anObject, NSUInteger idx, BOOL *stop) {
        return [filterSet containsObject: anObject];
    }];
    NSLog(@"Filtered: %@", [array objectsAtIndexes: indexSet]);
    
	NSLog(@"Case Insensitive Sorted: %@", [array sortedArrayUsingComparator: ^(id a, id b) { return [a caseInsensitiveCompare: b]; }]);
    
    return 0;
}
