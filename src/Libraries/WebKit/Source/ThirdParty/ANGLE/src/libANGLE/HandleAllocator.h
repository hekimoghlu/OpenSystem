/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// HandleAllocator.h: Defines the gl::HandleAllocator class, which is used to
// allocate GL handles.

#ifndef LIBANGLE_HANDLEALLOCATOR_H_
#define LIBANGLE_HANDLEALLOCATOR_H_

#include "common/angleutils.h"

#include "angle_gl.h"

namespace gl
{

class HandleAllocator final : angle::NonCopyable
{
  public:
    // Maximum handle = MAX_UINT-1
    HandleAllocator();
    // Specify maximum handle value. Used for testing.
    HandleAllocator(GLuint maximumHandleValue);

    ~HandleAllocator();

    void setBaseHandle(GLuint value);

    GLuint allocate();
    void release(GLuint handle);
    void reserve(GLuint handle);
    void reset();
    bool anyHandleAvailableForAllocation() const;

    void enableLogging(bool enabled);

  private:
    GLuint mBaseValue;
    GLuint mNextValue;
    const GLuint mMaxValue;

    // Represents an inclusive range [begin, end]
    struct HandleRange
    {
        HandleRange(GLuint beginIn, GLuint endIn) : begin(beginIn), end(endIn) {}

        GLuint begin;
        GLuint end;
    };

    struct HandleRangeComparator;

    // The freelist consists of never-allocated handles, stored
    // as ranges, and handles that were previously allocated and
    // released, stored in a heap.
    std::vector<HandleRange> mUnallocatedList;
    std::vector<GLuint> mReleasedList;

    bool mLoggingEnabled;
};

}  // namespace gl

#endif  // LIBANGLE_HANDLEALLOCATOR_H_
