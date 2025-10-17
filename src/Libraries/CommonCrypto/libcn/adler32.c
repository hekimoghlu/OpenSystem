/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#include "crc.h"

static const int adler_mod_value = 65521;

static uint64_t
adler32_setup() { return 0; }

static uint64_t
adler32_implementation(size_t len, const void *in, uint64_t __unused crc)
{
    uint32_t a = 1, b = 0;
    const uint8_t *bytes = in;
    
    for (size_t i = 0; i < len; i++) {
        a = (a + bytes[i]) % adler_mod_value;
        b = (b + a) % adler_mod_value;
    }
    return (b << 16) | a;
}

static uint64_t
adler32_final(size_t __unused length, uint64_t crc) { return crc; }


static uint64_t
adler32_oneshot(size_t len, const void *in)
{
    return adler32_implementation(len, in, 0);
}



const crcDescriptor adler32 = {
    .name = "adler-32",
    .defType = functions,
    .def.funcs.setup = adler32_setup,
    .def.funcs.update = adler32_implementation,
    .def.funcs.final = adler32_final,
    .def.funcs.oneshot = adler32_oneshot
};
