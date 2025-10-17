/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
#ifndef VPX_VPX_PORTS_STATIC_ASSERT_H_
#define VPX_VPX_PORTS_STATIC_ASSERT_H_

#if defined(_MSC_VER)
#define VPX_STATIC_ASSERT(boolexp)              \
  do {                                          \
    char vpx_static_assert[(boolexp) ? 1 : -1]; \
    (void)vpx_static_assert;                    \
  } while (0)
#else  // !_MSC_VER
#define VPX_STATIC_ASSERT(boolexp)                         \
  do {                                                     \
    struct {                                               \
      unsigned int vpx_static_assert : (boolexp) ? 1 : -1; \
    } vpx_static_assert;                                   \
    (void)vpx_static_assert;                               \
  } while (0)
#endif  // _MSC_VER

#endif  // VPX_VPX_PORTS_STATIC_ASSERT_H_
