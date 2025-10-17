/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// RenderTargetD3D.cpp: Implements serial handling for rx::RenderTargetD3D

#include "libANGLE/renderer/d3d/RenderTargetD3D.h"

namespace rx
{
unsigned int RenderTargetD3D::mCurrentSerial = 1;

RenderTargetD3D::RenderTargetD3D() : mSerial(issueSerials(1)) {}

RenderTargetD3D::~RenderTargetD3D() {}

unsigned int RenderTargetD3D::getSerial() const
{
    return mSerial;
}

unsigned int RenderTargetD3D::issueSerials(unsigned int count)
{
    unsigned int firstSerial = mCurrentSerial;
    mCurrentSerial += count;
    return firstSerial;
}

}  // namespace rx
