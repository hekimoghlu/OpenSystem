/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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
#include "swapyv12buffer.h"

void vp8_swap_yv12_buffer(YV12_BUFFER_CONFIG *new_frame,
                          YV12_BUFFER_CONFIG *last_frame) {
  unsigned char *temp;

  temp = last_frame->buffer_alloc;
  last_frame->buffer_alloc = new_frame->buffer_alloc;
  new_frame->buffer_alloc = temp;

  temp = last_frame->y_buffer;
  last_frame->y_buffer = new_frame->y_buffer;
  new_frame->y_buffer = temp;

  temp = last_frame->u_buffer;
  last_frame->u_buffer = new_frame->u_buffer;
  new_frame->u_buffer = temp;

  temp = last_frame->v_buffer;
  last_frame->v_buffer = new_frame->v_buffer;
  new_frame->v_buffer = temp;
}
