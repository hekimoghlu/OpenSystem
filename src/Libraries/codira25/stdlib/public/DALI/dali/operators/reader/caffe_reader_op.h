/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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

// Copyright (c) 2017-2018, 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_CAFFE_READER_OP_H_
#define DALI_OPERATORS_READER_CAFFE_READER_OP_H_

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/lmdb.h"
#include "dali/operators/reader/parser/caffe_parser.h"

namespace dali {

class CaffeReader : public DataReader<CPUBackend, Tensor<CPUBackend>, Tensor<CPUBackend>, true> {
 public:
  explicit CaffeReader(const OpSpec& spec)
  : DataReader<CPUBackend, Tensor<CPUBackend>, Tensor<CPUBackend>, true>(spec) {
    loader_ = InitLoader<LMDBLoader>(spec);
    parser_.reset(new CaffeParser(spec));
    this->SetInitialSnapshot();
  }

  void RunImpl(SampleWorkspace &ws) override {
    const auto& tensor = GetSample(ws.data_idx());
    ParseIfNeeded(tensor, &ws);
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, Tensor<CPUBackend>, Tensor<CPUBackend>, true);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_CAFFE_READER_OP_H_
