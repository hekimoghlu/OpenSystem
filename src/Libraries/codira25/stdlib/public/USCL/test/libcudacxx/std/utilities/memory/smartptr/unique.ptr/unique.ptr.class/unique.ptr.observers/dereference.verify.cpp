/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
// <memory>

// unique_ptr

// test op*()

#include <uscl/std/__memory_>

#include "test_macros.h"

void f()
{
  cuda::std::unique_ptr<int[]> p(new int(3));
  const cuda::std::unique_ptr<int[]>& cp = p;
  TEST_IGNORE_NODISCARD(*p); // expected-error-re {{indirection requires pointer operand ('cuda::std::unique_ptr<int{{[
                             // ]*}}[]>' invalid)}}
  TEST_IGNORE_NODISCARD(*cp); // expected-error-re {{indirection requires pointer operand ('const
                              // cuda::std::unique_ptr<int{{[ ]*}}[]>' invalid)}}
}
