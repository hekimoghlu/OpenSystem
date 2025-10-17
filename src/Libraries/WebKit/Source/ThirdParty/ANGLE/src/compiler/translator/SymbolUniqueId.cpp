/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 27, 2022.
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
// SymbolUniqueId.cpp: Encapsulates a unique id for a symbol.

#include "compiler/translator/SymbolUniqueId.h"

#include "compiler/translator/SymbolTable.h"

namespace sh
{

TSymbolUniqueId::TSymbolUniqueId(TSymbolTable *symbolTable) : mId(symbolTable->nextUniqueIdValue())
{}

TSymbolUniqueId::TSymbolUniqueId(const TSymbol &symbol) : mId(symbol.uniqueId().get()) {}

TSymbolUniqueId &TSymbolUniqueId::operator=(const TSymbolUniqueId &) = default;

bool TSymbolUniqueId::operator==(const TSymbolUniqueId &other) const
{
    return mId == other.mId;
}

bool TSymbolUniqueId::operator!=(const TSymbolUniqueId &other) const
{
    return !(*this == other);
}

}  // namespace sh
