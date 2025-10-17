/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// geometry_utils:
//   Helper library for generating certain sets of geometry.
//

#ifndef UTIL_GEOMETRY_UTILS_H
#define UTIL_GEOMETRY_UTILS_H

#include <cstddef>
#include <vector>

#include <GLES2/gl2.h>

#include "common/vector_utils.h"
#include "util/util_export.h"

struct ANGLE_UTIL_EXPORT SphereGeometry
{
    SphereGeometry();
    ~SphereGeometry();

    std::vector<angle::Vector3> positions;
    std::vector<angle::Vector3> normals;
    std::vector<GLushort> indices;
};

ANGLE_UTIL_EXPORT void CreateSphereGeometry(size_t sliceCount,
                                            float radius,
                                            SphereGeometry *result);

struct ANGLE_UTIL_EXPORT CubeGeometry
{
    CubeGeometry();
    ~CubeGeometry();

    std::vector<angle::Vector3> positions;
    std::vector<angle::Vector3> normals;
    std::vector<angle::Vector2> texcoords;
    std::vector<GLushort> indices;
};

ANGLE_UTIL_EXPORT void GenerateCubeGeometry(float radius, CubeGeometry *result);

#endif  // UTIL_GEOMETRY_UTILS_H
