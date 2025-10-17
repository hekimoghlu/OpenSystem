/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#ifndef COMMON_AUDIO_FIR_FILTER_FACTORY_H_
#define COMMON_AUDIO_FIR_FILTER_FACTORY_H_

#include <string.h>

namespace webrtc {

class FIRFilter;

// Creates a filter with the given coefficients. All initial state values will
// be zeros.
// The length of the chunks fed to the filter should never be greater than
// `max_input_length`. This is needed because, when vectorizing it is
// necessary to concatenate the input after the state, and resizing this array
// dynamically is expensive.
FIRFilter* CreateFirFilter(const float* coefficients,
                           size_t coefficients_length,
                           size_t max_input_length);

}  // namespace webrtc

#endif  // COMMON_AUDIO_FIR_FILTER_FACTORY_H_
