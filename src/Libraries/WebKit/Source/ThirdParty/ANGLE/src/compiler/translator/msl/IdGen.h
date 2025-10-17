/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_MSL_IDGEN_H_
#define COMPILER_TRANSLATOR_MSL_IDGEN_H_

#include "common/angleutils.h"
#include "compiler/translator/Name.h"

namespace sh
{

// For creating new fresh names.
// All names created are marked as SymbolType::AngleInternal.
class IdGen : angle::NonCopyable
{
  public:
    IdGen();

    Name createNewName(const ImmutableString &baseName);
    Name createNewName(const Name &baseName);
    Name createNewName(const char *baseName);
    Name createNewName(std::initializer_list<ImmutableString> baseNames);
    Name createNewName(std::initializer_list<Name> baseNames);
    Name createNewName(std::initializer_list<const char *> baseNames);
    Name createNewName();

  private:
    template <typename String, typename StringToImmutable>
    Name createNewName(size_t count, const String *baseNames, const StringToImmutable &toImmutable);

  private:
    unsigned mNext = 0;          // `unsigned` because of "%u" use in sprintf
    std::string mNewNameBuffer;  // reusable buffer to avoid tons of reallocations
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_MSL_IDGEN_H_
