/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
#include "rust_wrapper.h"


int ERR_GET_LIB_RUST(uint32_t packed_error) {
  return ERR_GET_LIB(packed_error);
}

int ERR_GET_REASON_RUST(uint32_t packed_error) {
  return ERR_GET_REASON(packed_error);
}

int ERR_GET_FUNC_RUST(uint32_t packed_error) {
  return ERR_GET_FUNC(packed_error);
}

void CBS_init_RUST(CBS *cbs, const uint8_t *data, size_t len) {
  CBS_init(cbs, data, len);
}

size_t CBS_len_RUST(const CBS *cbs) {
  return CBS_len(cbs);
}
