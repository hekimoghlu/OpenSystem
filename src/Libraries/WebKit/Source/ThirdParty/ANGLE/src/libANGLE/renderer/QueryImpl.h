/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// QueryImpl.h: Defines the abstract rx::QueryImpl class.

#ifndef LIBANGLE_RENDERER_QUERYIMPL_H_
#define LIBANGLE_RENDERER_QUERYIMPL_H_

#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"

namespace gl
{
class Context;
}  // namespace gl

namespace rx
{
class QueryImpl : angle::NonCopyable
{
  public:
    explicit QueryImpl(gl::QueryType type) : mType(type) {}
    virtual ~QueryImpl() {}

    virtual void onDestroy(const gl::Context *context);

    virtual angle::Result begin(const gl::Context *context)                              = 0;
    virtual angle::Result end(const gl::Context *context)                                = 0;
    virtual angle::Result queryCounter(const gl::Context *context)                       = 0;
    virtual angle::Result getResult(const gl::Context *context, GLint *params)           = 0;
    virtual angle::Result getResult(const gl::Context *context, GLuint *params)          = 0;
    virtual angle::Result getResult(const gl::Context *context, GLint64 *params)         = 0;
    virtual angle::Result getResult(const gl::Context *context, GLuint64 *params)        = 0;
    virtual angle::Result isResultAvailable(const gl::Context *context, bool *available) = 0;

    virtual angle::Result onLabelUpdate(const gl::Context *context);

    gl::QueryType getType() const { return mType; }

  protected:
    gl::QueryType mType;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_QUERYIMPL_H_
