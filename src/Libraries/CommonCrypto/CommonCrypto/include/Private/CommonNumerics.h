/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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
#ifndef COMMON_NUMERICS_H
#define COMMON_NUMERICS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
    
enum {
    kCNSuccess          = 0,
    kCNFailure          = 1,
    kCNParamError       = -4300,
    kCNBufferTooSmall   = -4301,
    kCNMemoryFailure    = -4302,
    kCNAlignmentError   = -4303,
    kCNDecodeError      = -4304,
    kCNUnimplemented    = -4305
};
typedef uint32_t CNStatus;
    
    
#ifdef __cplusplus
}
#endif


#endif /* COMMON_NUMERICS_H */
