/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 21, 2024.
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

#ifndef COMPILER_PREPROCESSOR_SOURCELOCATION_H_
#define COMPILER_PREPROCESSOR_SOURCELOCATION_H_

namespace angle
{

namespace pp
{

struct SourceLocation
{
    SourceLocation() : file(0), line(0) {}
    SourceLocation(int f, int l) : file(f), line(l) {}

    bool equals(const SourceLocation &other) const
    {
        return (file == other.file) && (line == other.line);
    }

    int file;
    int line;
};

inline bool operator==(const SourceLocation &lhs, const SourceLocation &rhs)
{
    return lhs.equals(rhs);
}

inline bool operator!=(const SourceLocation &lhs, const SourceLocation &rhs)
{
    return !lhs.equals(rhs);
}

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_SOURCELOCATION_H_
