/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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

#include <stdio.h>
#include <objc/runtime.h>
#import <Foundation/Foundation.h>

@interface TargetClass : NSObject
@end

@interface TargetClass(LoadedMethods)
- (void) m0;
- (void) m1;
- (void) m2;
- (void) m3;
- (void) m4;
- (void) m5;
- (void) m6;
- (void) m7;
- (void) m8;
- (void) m9;
- (void) m10;
- (void) m11;
- (void) m12;
- (void) m13;
- (void) m14;
- (void) m15;
@end

@interface TN:TargetClass
@end

@implementation TN
- (void) m1 { [super m1]; }
- (void) m3 { [self m1]; }

- (void) m2
{
    [self willChangeValueForKey: @"m4"];
    [self didChangeValueForKey: @"m4"];
}

- (void)observeValueForKeyPath:(NSString *) keyPath
		      ofObject:(id)object
			change:(NSDictionary *)change
		       context:(void *)context
{
    // suppress warning
    (void)keyPath;
    (void)object;
    (void)change;
    (void)context;
}
@end

@implementation TargetClass(LoadedMethods)
- (void) m0 { ; }
- (void) m1 { ; }
- (void) m2 { ; }
- (void) m3 { ; }
- (void) m4 { ; }
- (void) m5 { ; }
- (void) m6 { ; }
- (void) m7 { ; }
- (void) m8 { ; }
- (void) m9 { ; }
- (void) m10 { ; }
- (void) m11 { ; }
- (void) m12 { ; }
- (void) m13 { ; }
- (void) m14 { ; }
- (void) m15 { ; }
@end
