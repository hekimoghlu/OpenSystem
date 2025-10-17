/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// QueryNULL.h:
//    Defines the class interface for QueryNULL, implementing QueryImpl.
//

#ifndef LIBANGLE_RENDERER_NULL_QUERYNULL_H_
#define LIBANGLE_RENDERER_NULL_QUERYNULL_H_

#include "libANGLE/renderer/QueryImpl.h"

namespace rx
{

class QueryNULL : public QueryImpl
{
  public:
    QueryNULL(gl::QueryType type);
    ~QueryNULL() override;

    angle::Result begin(const gl::Context *context) override;
    angle::Result end(const gl::Context *context) override;
    angle::Result queryCounter(const gl::Context *context) override;
    angle::Result getResult(const gl::Context *context, GLint *params) override;
    angle::Result getResult(const gl::Context *context, GLuint *params) override;
    angle::Result getResult(const gl::Context *context, GLint64 *params) override;
    angle::Result getResult(const gl::Context *context, GLuint64 *params) override;
    angle::Result isResultAvailable(const gl::Context *context, bool *available) override;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_NULL_QUERYNULL_H_
