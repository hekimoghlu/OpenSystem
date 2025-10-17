/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
#include "config.h"
#include "GlobalContextWithFinalizerTest.h"

#include "JavaScript.h"
#include <stdio.h>

static bool failed = true;

static void finalize(JSObjectRef)
{
    failed = false;
}

int testGlobalContextWithFinalizer()
{
    JSClassDefinition def = kJSClassDefinitionEmpty;
    def.className = "testClass";
    def.finalize = finalize;
    JSClassRef classRef = JSClassCreate(&def);
    
    JSGlobalContextRef ref = JSGlobalContextCreateInGroup(nullptr, classRef);
    JSGlobalContextRelease(ref);
    JSClassRelease(classRef);

    if (failed)
        printf("FAIL: JSGlobalContextRef did not call its JSClassRef finalizer.\n");
    else
        printf("PASS: JSGlobalContextRef called its JSClassRef finalizer as expected.\n");

    return failed;
}
