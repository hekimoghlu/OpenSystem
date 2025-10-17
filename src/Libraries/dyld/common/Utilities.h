/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#ifndef Utilities_h
#define Utilities_h

#include <stdio.h>
#include <stdint.h>

#include "Defines.h"

namespace dyld4 {

namespace Utils {

VIS_HIDDEN
const char* strrstr(const char* str, const char* sub);

VIS_HIDDEN
size_t concatenatePaths(char *path, const char *suffix, size_t pathsize);

}; /* namespace Utils */

}; /* namespace dyld4 */


// escape a cstring literal, output buffer is always null terminated and parameter `end` will point to the null terminator if given
VIS_HIDDEN
void escapeCStringLiteral(const char* s, char* b, size_t bufferLength, char**end=nullptr);


#if __has_feature(ptrauth_calls)
// PAC sign an arm64e pointer
VIS_HIDDEN
uint64_t signPointer(uint64_t unsignedAddr, void* loc, bool addrDiv, uint16_t diversity, ptrauth_key key);
#endif

#endif /* Utilities_h */
