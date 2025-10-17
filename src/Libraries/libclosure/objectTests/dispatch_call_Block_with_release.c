/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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
#include <Block.h>
#include "test.h"

// TEST_CONFIG

void callsomething(const char *format __unused, int argument __unused) {
    asm("");
}

void
dispatch_call_Block_with_release2(void *block)
{
        void (^b)(void) = (void (^)(void))block;
        b();
        Block_release(b);
}

int main(int argc, char *argv[] __unused) {
     void (^b1)(void) = ^{ callsomething("argc is %d\n", argc); };
     void (^b2)(void) = ^{ callsomething("hellow world\n", 0); }; // global block now

     dispatch_call_Block_with_release2(Block_copy(b1));
     dispatch_call_Block_with_release2(Block_copy(b2));

     succeed(__FILE__);
}
