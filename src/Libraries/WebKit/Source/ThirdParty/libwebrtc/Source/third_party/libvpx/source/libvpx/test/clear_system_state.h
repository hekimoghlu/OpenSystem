/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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
#ifndef VPX_TEST_CLEAR_SYSTEM_STATE_H_
#define VPX_TEST_CLEAR_SYSTEM_STATE_H_

#include "./vpx_config.h"
#include "vpx_ports/system_state.h"

namespace libvpx_test {

// Reset system to a known state. This function should be used for all non-API
// test cases.
inline void ClearSystemState() { vpx_clear_system_state(); }

}  // namespace libvpx_test
#endif  // VPX_TEST_CLEAR_SYSTEM_STATE_H_
