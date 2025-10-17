/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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

// TEST_CONFIG MEM=mrc LANGUAGE=objective-c ARCH=!arm64e,!arm64
/*
TEST_RUN_OUTPUT
[\S\s]*0 leaks for 0 total leaked bytes[\S\s]*
END
*/

#include "test.h"
#include "testroot.i"

#include <spawn.h>
#include <stdio.h>

void noopIMP(id self __unused, SEL _cmd __unused) {}

id test(int n, int methodCount) {
    char *name;
    asprintf(&name, "TestClass%d", n);
    Class c = objc_allocateClassPair([TestRoot class], name, 0);
    free(name);
    
    SEL *sels = malloc(methodCount * sizeof(*sels));
    for(int i = 0; i < methodCount; i++) {
        asprintf(&name, "selector%d", i);
        sels[i] = sel_getUid(name);
        free(name);
    }
    
    for(int i = 0; i < methodCount; i++) {
        class_addMethod(c, sels[i], (IMP)noopIMP, "v@:");
    }
    
    objc_registerClassPair(c);
    
    id obj = [[c alloc] init];
    for (int i = 0; i < methodCount; i++) {
        ((void (*)(id, SEL))objc_msgSend)(obj, sels[i]);
    }
    free(sels);
    return obj;
}

int main()
{
    int classCount = 16;
    id *objs = malloc(classCount * sizeof(*objs));
    for (int i = 0; i < classCount; i++) {
        objs[i] = test(i, 1 << i);
    }
    
    char *pidstr;
    int result = asprintf(&pidstr, "%u", getpid());
    testassert(result);
    
    extern char **environ;
    char *argv[] = { "/usr/bin/leaks", pidstr, NULL };
    pid_t pid;
    result = posix_spawn(&pid, "/usr/bin/leaks", NULL, NULL, argv, environ);
    if (result) {
        perror("posix_spawn");
        exit(1);
    }
    wait4(pid, NULL, 0, NULL);

    // Clean up. Otherwise leaks can end up seeing this as a leak, oddly enough.
    for (int i = 0; i < classCount; i++) {
       [objs[i] release];
    }
    free(objs);
}
