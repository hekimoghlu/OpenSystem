/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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

static int globalCount = 0;

@interface Foo : NSObject {
    int ivarCount;
}

- (void) incrementCount;
@end

@implementation Foo 
- (void) incrementCount
{
    int oldValue = ivarCount;
    
    void (^incrementBlock)()  = ^(){ivarCount++;};
    incrementBlock();
    if( (oldValue+1) != ivarCount )
        NSLog(@"Hey, man.  ivar was not incremented as expected.  %d %d", oldValue, ivarCount);
    }
@end


int main (int argc, const char * argv[]) {
    int localCount = 0;

    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

    [[[[Foo alloc] init] autorelease] incrementCount];
    
    void (^incrementLocal)() = ^(){
        |localCount| // comment this out for an exciting compilation error on the next line (that is correct)
        localCount++;
    };
    
    incrementLocal();
    if( localCount != 1 )
        NSLog(@"Hey, man.  localCount was not incremented as expected.  %d", localCount);
    
    void (^incrementGlobal)() = ^() {
        |globalCount| // this should not be necessary
        globalCount++;
    };
    incrementGlobal();
    if( globalCount != 1 )
        NSLog(@"Hey, man.  globalCount was not incremented as expected.  %d", globalCount);

    [pool drain];
    return 0;
}
