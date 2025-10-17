/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
#include "test.h"
#include <objc/runtime.h>
#include <objc/objc-auto.h>
#include <dlfcn.h>
#include <unistd.h>
#include <pthread.h>
#include <Foundation/Foundation.h>

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

@implementation TargetClass
- (void) m0 { fail("shoulda been loaded!"); }
- (void) m1 { fail("shoulda been loaded!"); }
- (void) m2 { fail("shoulda been loaded!"); }
- (void) m3 { fail("shoulda been loaded!"); }
- (void) m4 { fail("shoulda been loaded!"); }
- (void) m5 { fail("shoulda been loaded!"); }
- (void) m6 { fail("shoulda been loaded!"); }
@end

void *threadFun(void *aTargetClassName) {
    const char *className = (const char *)aTargetClassName;

    PUSH_POOL {
        
        Class targetSubclass = objc_getClass(className);
        testassert(targetSubclass);
        
        id target = [targetSubclass new];
        testassert(target);
        
        while(1) {
            [target m0];
            RETAIN(target);
            [target addObserver: target forKeyPath: @"m3" options: 0 context: NULL];
            [target addObserver: target forKeyPath: @"m4" options: 0 context: NULL];
            [target m1];
            RELEASE_VALUE(target);
            [target m2];
            AUTORELEASE(target);
            [target m3];
            RETAIN(target);
            [target removeObserver: target forKeyPath: @"m4"];
            [target addObserver: target forKeyPath: @"m5" options: 0 context: NULL];
            [target m4];
            RETAIN(target);
            [target m5];
            AUTORELEASE(target);
            [target m6];
            [target m7];
            [target m8];
            [target m9];
            [target m10];
            [target m11];
            [target m12];
            [target m13];
            [target m14];
            [target m15];
            [target removeObserver: target forKeyPath: @"m3"];
            [target removeObserver: target forKeyPath: @"m5"];
        }
    } POP_POOL;
    return NULL;
}

int main()
{
    int i;

    void *dylib;

    for(i=1; i<16; i++) {
	pthread_t t;
	char dlName[100];
	snprintf(dlName, sizeof(dlName), "cc%d.bundle", i);
	dylib = dlopen(dlName, RTLD_LAZY);
	char className[100];
	snprintf(className, sizeof(className), "cc%d", i);
	pthread_create(&t, NULL, threadFun, strdup(className));
	testassert(dylib);
    }
    sleep(1);

    succeed(__FILE__);
}
