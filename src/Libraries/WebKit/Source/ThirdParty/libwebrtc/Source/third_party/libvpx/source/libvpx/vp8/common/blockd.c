/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
#include "blockd.h"
#include "vpx_mem/vpx_mem.h"

const unsigned char vp8_block2left[25] = { 0, 0, 0, 0, 1, 1, 1, 1, 2,
                                           2, 2, 2, 3, 3, 3, 3, 4, 4,
                                           5, 5, 6, 6, 7, 7, 8 };
const unsigned char vp8_block2above[25] = { 0, 1, 2, 3, 0, 1, 2, 3, 0,
                                            1, 2, 3, 0, 1, 2, 3, 4, 5,
                                            4, 5, 6, 7, 6, 7, 8 };
