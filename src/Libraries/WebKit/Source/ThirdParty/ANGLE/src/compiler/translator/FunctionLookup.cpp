/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
// FunctionLookup.cpp: Used for storing function calls that have not yet been resolved during
// parsing.
//

#include "compiler/translator/FunctionLookup.h"
#include "compiler/translator/ImmutableStringBuilder.h"

namespace sh
{

namespace
{

const char kFunctionMangledNameSeparator = '(';

constexpr const ImmutableString kEmptyName("");

}  // anonymous namespace

TFunctionLookup::TFunctionLookup(const ImmutableString &name,
                                 const TType *constructorType,
                                 const TSymbol *symbol)
    : mName(name), mConstructorType(constructorType), mThisNode(nullptr), mSymbol(symbol)
{}

// static
TFunctionLookup *TFunctionLookup::CreateConstructor(const TType *type)
{
    ASSERT(type != nullptr);
    return new TFunctionLookup(kEmptyName, type, nullptr);
}

// static
TFunctionLookup *TFunctionLookup::CreateFunctionCall(const ImmutableString &name,
                                                     const TSymbol *symbol)
{
    ASSERT(name != "");
    return new TFunctionLookup(name, nullptr, symbol);
}

const ImmutableString &TFunctionLookup::name() const
{
    return mName;
}

ImmutableString TFunctionLookup::getMangledName() const
{
    return GetMangledName(mName.data(), mArguments);
}

ImmutableString TFunctionLookup::GetMangledName(const char *functionName,
                                                const TIntermSequence &arguments)
{
    std::string newName(functionName);
    newName += kFunctionMangledNameSeparator;

    for (TIntermNode *argument : arguments)
    {
        newName += argument->getAsTyped()->getType().getMangledName();
    }
    return ImmutableString(newName);
}

bool TFunctionLookup::isConstructor() const
{
    return mConstructorType != nullptr;
}

const TType &TFunctionLookup::constructorType() const
{
    return *mConstructorType;
}

void TFunctionLookup::setThisNode(TIntermTyped *thisNode)
{
    mThisNode = thisNode;
}

TIntermTyped *TFunctionLookup::thisNode() const
{
    return mThisNode;
}

void TFunctionLookup::addArgument(TIntermTyped *argument)
{
    mArguments.push_back(argument);
}

TIntermSequence &TFunctionLookup::arguments()
{
    return mArguments;
}

const TSymbol *TFunctionLookup::symbol() const
{
    return mSymbol;
}

}  // namespace sh
