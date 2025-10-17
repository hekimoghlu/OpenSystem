/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DrawCallPerfParams.cpp:
//   Parametrization for performance tests for ANGLE draw call overhead.
//

#include "DrawCallPerfParams.h"

#include <sstream>

DrawCallPerfParams::DrawCallPerfParams()
{
    majorVersion = 2;
    minorVersion = 0;
    windowWidth  = 64;
    windowHeight = 64;

// Lower the iteration count in debug.
#if !defined(NDEBUG)
    iterationsPerStep = 100;
#else
    iterationsPerStep = 20000;
#endif
    runTimeSeconds = 10.0;
    numTris        = 1;
}

DrawCallPerfParams::~DrawCallPerfParams() = default;

std::string DrawCallPerfParams::story() const
{
    std::stringstream strstr;

    strstr << RenderTestParams::story();

    return strstr.str();
}
