/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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

// Query.h: Defines the gl::Query class

#ifndef LIBANGLE_QUERY_H_
#define LIBANGLE_QUERY_H_

#include "common/PackedEnums.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"

#include "common/angleutils.h"

#include "angle_gl.h"

namespace rx
{
class GLImplFactory;
class QueryImpl;
}  // namespace rx

namespace gl
{

class Query final : public RefCountObject<QueryID>, public LabeledObject
{
  public:
    Query(rx::GLImplFactory *factory, QueryType type, QueryID id);
    ~Query() override;
    void onDestroy(const Context *context) override;

    angle::Result setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    angle::Result begin(const Context *context);
    angle::Result end(const Context *context);
    angle::Result queryCounter(const Context *context);
    angle::Result getResult(const Context *context, GLint *params);
    angle::Result getResult(const Context *context, GLuint *params);
    angle::Result getResult(const Context *context, GLint64 *params);
    angle::Result getResult(const Context *context, GLuint64 *params);
    angle::Result isResultAvailable(const Context *context, bool *available);

    QueryType getType() const;

    rx::QueryImpl *getImplementation() const;

  private:
    rx::QueryImpl *mQuery;

    std::string mLabel;
};
}  // namespace gl

#endif  // LIBANGLE_QUERY_H_
