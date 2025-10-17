/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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
// Declarator.h:
//   Declarator type for parsing structure field declarators.

#ifndef COMPILER_TRANSLATOR_DECLARATOR_H_
#define COMPILER_TRANSLATOR_DECLARATOR_H_

#include "compiler/translator/Common.h"
#include "compiler/translator/ImmutableString.h"

namespace sh
{

// Declarator like "a[2][4]". Only used for parsing structure field declarators.
class TDeclarator : angle::NonCopyable
{
  public:
    POOL_ALLOCATOR_NEW_DELETE
    TDeclarator(const ImmutableString &name, const TSourceLoc &line);

    TDeclarator(const ImmutableString &name,
                const TVector<unsigned int> *arraySizes,
                const TSourceLoc &line);

    const ImmutableString &name() const { return mName; }

    bool isArray() const;
    const TVector<unsigned int> *arraySizes() const { return mArraySizes; }

    const TSourceLoc &line() const { return mLine; }

  private:
    const ImmutableString mName;

    // Outermost array size is stored at the end of the vector.
    const TVector<unsigned int> *const mArraySizes;

    const TSourceLoc mLine;
};

using TDeclaratorList = TVector<TDeclarator *>;

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_DECLARATOR_H_
