/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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

#ifndef ABSL_DEBUGGING_INTERNAL_STACKTRACE_UNIMPLEMENTED_INL_H_
#define ABSL_DEBUGGING_INTERNAL_STACKTRACE_UNIMPLEMENTED_INL_H_

template <bool IS_STACK_FRAMES, bool IS_WITH_CONTEXT>
static int UnwindImpl(void** /* result */, int* /* sizes */,
                      int /* max_depth */, int /* skip_count */,
                      const void* /* ucp */, int *min_dropped_frames) {
  if (min_dropped_frames != nullptr) {
    *min_dropped_frames = 0;
  }
  return 0;
}

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace debugging_internal {
bool StackTraceWorksForTest() {
  return false;
}
}  // namespace debugging_internal
ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_DEBUGGING_INTERNAL_STACKTRACE_UNIMPLEMENTED_INL_H_
