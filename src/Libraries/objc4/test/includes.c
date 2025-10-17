/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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

// TEST_CONFIG MEM=mrc,arc LANGUAGE=c,c++,objc,objc++

// Verify that all headers can be included in any language.
// See also test/include-warnings.c which checks for warnings in these headers.
// See also test/includes-objc2.c which checks for safety even if 
// the client is C code that defined __OBJC2__.

#ifndef NAME
#define NAME "includes.c"
#endif

#include <objc/objc.h>

#include <objc/List.h>
#include <objc/NSObjCRuntime.h>
#include <objc/NSObject.h>
#include <objc/Object.h>
#include <objc/Protocol.h>
#include <objc/message.h>
#include <objc/objc-api.h>
#include <objc/objc-auto.h>
#include <objc/objc-class.h>
#include <objc/objc-exception.h>
#include <objc/objc-load.h>
#include <objc/objc-runtime.h>
#include <objc/objc-sync.h>
#include <objc/runtime.h>

#include <objc/objc-abi.h>
#include <objc/objc-gdb.h>
#include <objc/objc-internal.h>

#if TARGET_OS_OSX
#include <objc/hashtable.h>
#include <objc/hashtable2.h>
#include <objc/maptable.h>
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include "test.h"
#pragma clang diagnostic pop

int main(int argc __unused, char **argv __unused)
{
    succeed(NAME);
}
