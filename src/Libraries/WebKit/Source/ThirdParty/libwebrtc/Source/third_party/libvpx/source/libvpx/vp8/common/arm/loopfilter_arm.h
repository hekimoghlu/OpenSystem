/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#ifndef VPX_VP8_COMMON_ARM_LOOPFILTER_ARM_H_
#define VPX_VP8_COMMON_ARM_LOOPFILTER_ARM_H_

typedef void loopfilter_y_neon(unsigned char *src, int pitch,
                               unsigned char blimit, unsigned char limit,
                               unsigned char thresh);
typedef void loopfilter_uv_neon(unsigned char *u, int pitch,
                                unsigned char blimit, unsigned char limit,
                                unsigned char thresh, unsigned char *v);

loopfilter_y_neon vp8_loop_filter_horizontal_edge_y_neon;
loopfilter_y_neon vp8_loop_filter_vertical_edge_y_neon;
loopfilter_uv_neon vp8_loop_filter_horizontal_edge_uv_neon;
loopfilter_uv_neon vp8_loop_filter_vertical_edge_uv_neon;

loopfilter_y_neon vp8_mbloop_filter_horizontal_edge_y_neon;
loopfilter_y_neon vp8_mbloop_filter_vertical_edge_y_neon;
loopfilter_uv_neon vp8_mbloop_filter_horizontal_edge_uv_neon;
loopfilter_uv_neon vp8_mbloop_filter_vertical_edge_uv_neon;

#endif  // VPX_VP8_COMMON_ARM_LOOPFILTER_ARM_H_
