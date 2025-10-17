/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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

#ifndef COMPILER_TRANSLATOR_NAME_H_
#define COMPILER_TRANSLATOR_NAME_H_

#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/InfoSink.h"
#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/Types.h"

namespace sh
{

constexpr char kAngleInternalPrefix[] = "ANGLE";

// Represents the name of a symbol.
class Name
{
  public:
    constexpr Name(const Name &) = default;

    constexpr Name() : Name(kEmptyImmutableString, SymbolType::Empty) {}

    constexpr Name(ImmutableString rawName, SymbolType symbolType)
        : mRawName(rawName), mSymbolType(symbolType)
    {
        ASSERT(rawName.empty() == (symbolType == SymbolType::Empty));
    }

    explicit constexpr Name(const char *rawName, SymbolType symbolType = SymbolType::AngleInternal)
        : Name(ImmutableString(rawName), symbolType)
    {}

    Name(const std::string &rawName, SymbolType symbolType)
        : Name(ImmutableString(rawName), symbolType)
    {}

    explicit Name(const TField &field);
    explicit Name(const TSymbol &symbol);

    Name &operator=(const Name &) = default;
    bool operator==(const Name &other) const;
    bool operator!=(const Name &other) const;
    bool operator<(const Name &other) const;

    constexpr const ImmutableString &rawName() const { return mRawName; }
    constexpr SymbolType symbolType() const { return mSymbolType; }

    bool empty() const;
    bool beginsWith(const Name &prefix) const;

    void emit(TInfoSinkBase &out) const;

  private:
    ImmutableString mRawName;
    SymbolType mSymbolType;
    template <typename T>
    void emitImpl(T &out) const;
    friend std::ostream &operator<<(std::ostream &os, const sh::Name &name);
};

constexpr Name kBaseInstanceName = Name("baseInstance");

[[nodiscard]] bool ExpressionContainsName(const Name &name, TIntermTyped &node);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_MSL_NAME_H_
