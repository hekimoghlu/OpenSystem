/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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

// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include "dali/pipeline/data/tensor.h"
#include "dali/test/tensor_list_wrapper.h"

namespace dali {
namespace testing {

TEST(TensorListWrapper, EmptyWrapperBoolOperator) {
  TensorListWrapper tw;
  EXPECT_FALSE(tw);
}


TEST(TensorListWrapper, HasCpu) {
  TensorList<CPUBackend> tl;
  TensorListWrapper tw(&tl);
  EXPECT_TRUE(tw.has_cpu());
  EXPECT_FALSE(tw.has_gpu());
}


TEST(TensorListWrapper, HasGpu) {
  TensorList<GPUBackend> tl;
  TensorListWrapper tw(&tl);
  EXPECT_TRUE(tw.has_gpu());
  EXPECT_FALSE(tw.has_cpu());
}

}  // namespace testing
}  // namespace dali
