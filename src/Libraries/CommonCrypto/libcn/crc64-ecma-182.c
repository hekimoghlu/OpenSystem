/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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

const crcDescriptor crc64_ecma_182 = {
    .name = "crc64-ecma-182",
    .defType = model,
    .def.parms.width = 8,
    .def.parms.poly = 0x42F0E1EBA9EA3693ULL,
    .def.parms.initial_value = 0xffffffffffffffffULL,
    .def.parms.final_xor = 0xffffffffffffffffULL,
    .def.parms.weak_check = 0x62EC59E3F1A4F00AULL,
    .def.parms.reflect_reverse = NO_REFLECT_REVERSE,
};
