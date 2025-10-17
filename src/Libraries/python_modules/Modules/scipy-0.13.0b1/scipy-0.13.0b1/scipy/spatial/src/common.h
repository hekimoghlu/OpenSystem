/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#ifndef _CLUSTER_COMMON_H
#define _CLUSTER_COMMON_H

#define CPY_MAX(_x, _y) ((_x > _y) ? (_x) : (_y))
#define CPY_MIN(_x, _y) ((_x < _y) ? (_x) : (_y))

#define NCHOOSE2(_n) ((_n)*(_n-1)/2)

#define CPY_BITS_PER_CHAR (sizeof(unsigned char) * 8)
#define CPY_FLAG_ARRAY_SIZE_BYTES(num_bits) (CPY_CEIL_DIV((num_bits), \
                                                          CPY_BITS_PER_CHAR))
#define CPY_GET_BIT(_xx, i) (((_xx)[(i) / CPY_BITS_PER_CHAR] >> \
                             ((CPY_BITS_PER_CHAR-1) - \
                              ((i) % CPY_BITS_PER_CHAR))) & 0x1)
#define CPY_SET_BIT(_xx, i) ((_xx)[(i) / CPY_BITS_PER_CHAR] |= \
                              ((0x1) << ((CPY_BITS_PER_CHAR-1) \
                                         -((i) % CPY_BITS_PER_CHAR))))
#define CPY_CLEAR_BIT(_xx, i) ((_xx)[(i) / CPY_BITS_PER_CHAR] &= \
                              ~((0x1) << ((CPY_BITS_PER_CHAR-1) \
                                         -((i) % CPY_BITS_PER_CHAR))))

#ifndef CPY_CEIL_DIV
#define CPY_CEIL_DIV(x, y) ((((double)x)/(double)y) == \
                            ((double)((x)/(y))) ? ((x)/(y)) : ((x)/(y) + 1))
#endif


#ifdef CPY_DEBUG
#define CPY_DEBUG_MSG(...) fprintf(stderr, __VA_ARGS__)
#else
#define CPY_DEBUG_MSG(...)
#endif

#endif
