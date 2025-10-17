/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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
// SymbolUniqueId.h: Encapsulates a unique id for a symbol.

#ifndef COMPILER_TRANSLATOR_SYMBOLUNIQUEID_H_
#define COMPILER_TRANSLATOR_SYMBOLUNIQUEID_H_

#include "compiler/translator/Common.h"

namespace sh
{

class TSymbolTable;
class TSymbol;

class TSymbolUniqueId
{
  public:
    POOL_ALLOCATOR_NEW_DELETE
    explicit TSymbolUniqueId(const TSymbol &symbol);
    constexpr TSymbolUniqueId(const TSymbolUniqueId &) = default;
    TSymbolUniqueId &operator=(const TSymbolUniqueId &);
    bool operator==(const TSymbolUniqueId &) const;
    bool operator!=(const TSymbolUniqueId &) const;

    constexpr int get() const { return mId; }

  private:
    friend class TSymbolTable;
    explicit TSymbolUniqueId(TSymbolTable *symbolTable);

    friend class BuiltInId;
    constexpr TSymbolUniqueId(int staticId) : mId(staticId) {}

    int mId;
};

enum class SymbolType : uint8_t
{
    BuiltIn,
    UserDefined,
    AngleInternal,
    Empty  // Meaning symbol without a name.
};

enum class SymbolClass : uint8_t
{
    Function,
    Variable,
    Struct,
    InterfaceBlock
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_SYMBOLUNIQUEID_H_
