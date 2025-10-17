/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FunctionLookup.h: Used for storing function calls that have not yet been resolved during parsing.
//

#ifndef COMPILER_TRANSLATOR_FUNCTIONLOOKUP_H_
#define COMPILER_TRANSLATOR_FUNCTIONLOOKUP_H_

#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/IntermNode.h"

namespace sh
{

// A function look-up.
class TFunctionLookup : angle::NonCopyable
{
  public:
    POOL_ALLOCATOR_NEW_DELETE

    static TFunctionLookup *CreateConstructor(const TType *type);
    static TFunctionLookup *CreateFunctionCall(const ImmutableString &name, const TSymbol *symbol);

    const ImmutableString &name() const;
    ImmutableString getMangledName() const;
    static ImmutableString GetMangledName(const char *functionName,
                                          const TIntermSequence &arguments);

    bool isConstructor() const;
    const TType &constructorType() const;

    void setThisNode(TIntermTyped *thisNode);
    TIntermTyped *thisNode() const;

    void addArgument(TIntermTyped *argument);
    TIntermSequence &arguments();

    // Symbol looked up in the lexical phase using only the name of the function.
    // This does not necessarily correspond to the correct overloaded function.
    const TSymbol *symbol() const;

  private:
    TFunctionLookup(const ImmutableString &name,
                    const TType *constructorType,
                    const TSymbol *symbol);

    const ImmutableString mName;
    const TType *const mConstructorType;
    TIntermTyped *mThisNode;
    TIntermSequence mArguments;
    const TSymbol *mSymbol;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_FUNCTIONLOOKUP_H_
