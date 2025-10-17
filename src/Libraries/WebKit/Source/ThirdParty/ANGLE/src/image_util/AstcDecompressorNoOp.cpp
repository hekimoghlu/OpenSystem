/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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

//
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// AstcDecompressorNoOp.cpp: No-op implementation if support for ASTC textures wasn't enabled

#include "image_util/AstcDecompressor.h"

namespace angle
{

namespace
{

class AstcDecompressorNoOp : public AstcDecompressor
{
  public:
    bool available() const override { return false; }

    int32_t decompress(std::shared_ptr<WorkerThreadPool> singleThreadPool,
                       std::shared_ptr<WorkerThreadPool> multiThreadPool,
                       uint32_t imgWidth,
                       uint32_t imgHeight,
                       uint32_t blockWidth,
                       uint32_t blockHeight,
                       const uint8_t *astcData,
                       size_t astcDataLength,
                       uint8_t *output) override
    {
        return -1;
    }

    const char *getStatusString(int32_t statusCode) const override
    {
        return "ASTC CPU decomp not available";
    }
};

}  // namespace

AstcDecompressor &AstcDecompressor::get()
{
    static AstcDecompressorNoOp *instance = new AstcDecompressorNoOp();
    return *instance;
}

}  // namespace angle
