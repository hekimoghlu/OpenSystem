/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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

// RUN: %empty-directory(%t)
// RUN: xcrun -sdk %target-sdk-name %clang -c -arch %target-cpu %s -o %t/SegmentAlignment.o
// RUN: %target-build-language %S/Inputs/SegmentAlignment.code -Xlinker %t/SegmentAlignment.o -o %t/a.out
// RUN: %target-run %t/a.out | %FileCheck %s
// REQUIRES: executable_test
// REQUIRES: CPU=armv7

// Verify 16K segment alignment on 32-bit iOS device.
// The linker sets this automatically on iOS 8+,
// but we deploy to iOS 7.

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

#if __LP64__
#define HeaderType struct mach_header_64
#else
#define HeaderType struct mach_header
#endif

// SegmentAlignment.code import SpriteKit and calls Test().
void Test(void)
{
    for (int i = 0; i < _dyld_image_count(); i++) {
        const char *name = _dyld_get_image_name(i);
        if (strstr(name, "liblanguage") == 0) continue;

        unsigned long size;
        const struct mach_header *mhdr = _dyld_get_image_header(i);
        uint8_t *textAddress =
            getsegmentdata((HeaderType *)mhdr, "__TEXT", &size);
        uint8_t *dataAddress =
            getsegmentdata((HeaderType *)mhdr, "__DATA", &size);

        printf("%s %p %p\n", name, textAddress, dataAddress);
        assert((uintptr_t)textAddress % 0x4000 == 0);
        assert((uintptr_t)dataAddress % 0x4000 == 0);
    }

    printf("Flawless victory\n");
    // CHECK-DAG: liblanguageSpriteKit.dylib
    // CHECK-DAG: liblanguageUIKit.dylib
    // CHECK-DAG: liblanguageCore.dylib
    // CHECK: Flawless victory
}
