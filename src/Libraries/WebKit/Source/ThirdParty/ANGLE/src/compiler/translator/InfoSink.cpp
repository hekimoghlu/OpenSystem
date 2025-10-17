/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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

#include "compiler/translator/InfoSink.h"

#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/Types.h"

namespace sh
{

void TInfoSinkBase::prefix(Severity severity)
{
    switch (severity)
    {
        case SH_WARNING:
            sink.append("WARNING: ");
            break;
        case SH_ERROR:
            sink.append("ERROR: ");
            break;
        default:
            sink.append("UNKOWN ERROR: ");
            break;
    }
}

TInfoSinkBase &TInfoSinkBase::operator<<(const ImmutableString &str)
{
    sink.append(str.data());
    return *this;
}

TInfoSinkBase &TInfoSinkBase::operator<<(const TType &type)
{
    if (type.isInvariant())
        sink.append("invariant ");
    if (type.getQualifier() != EvqTemporary && type.getQualifier() != EvqGlobal)
    {
        sink.append(type.getQualifierString());
        sink.append(" ");
    }
    if (type.getPrecision() != EbpUndefined)
    {
        sink.append(type.getPrecisionString());
        sink.append(" ");
    }

    const TMemoryQualifier &memoryQualifier = type.getMemoryQualifier();
    if (memoryQualifier.readonly)
    {
        sink.append("readonly ");
    }
    if (memoryQualifier.writeonly)
    {
        sink.append("writeonly ");
    }
    if (memoryQualifier.coherent)
    {
        sink.append("coherent ");
    }
    if (memoryQualifier.restrictQualifier)
    {
        sink.append("restrict ");
    }
    if (memoryQualifier.volatileQualifier)
    {
        sink.append("volatile ");
    }

    if (type.isArray())
    {
        for (auto arraySizeIter = type.getArraySizes().rbegin();
             arraySizeIter != type.getArraySizes().rend(); ++arraySizeIter)
        {
            *this << "array[" << (*arraySizeIter) << "] of ";
        }
    }
    if (type.isMatrix())
    {
        *this << static_cast<uint32_t>(type.getCols()) << "X"
              << static_cast<uint32_t>(type.getRows()) << " matrix of ";
    }
    else if (type.isVector())
        *this << static_cast<uint32_t>(type.getNominalSize()) << "-component vector of ";

    sink.append(type.getBasicString());

    if (type.getStruct() != nullptr)
    {
        *this << ' ' << static_cast<const TSymbol &>(*type.getStruct());
        if (type.isStructSpecifier())
        {
            *this << " (specifier)";
        }
    }

    return *this;
}

TInfoSinkBase &TInfoSinkBase::operator<<(const TSymbol &symbol)
{
    switch (symbol.symbolType())
    {
        case (SymbolType::BuiltIn):
            *this << symbol.name();
            break;
        case (SymbolType::Empty):
            *this << "''";
            break;
        case (SymbolType::AngleInternal):
            *this << '#' << symbol.name();
            break;
        case (SymbolType::UserDefined):
            *this << '\'' << symbol.name() << '\'';
            break;
    }
    *this << " (symbol id " << symbol.uniqueId().get() << ")";
    return *this;
}

void TInfoSinkBase::location(int file, int line)
{
    TPersistStringStream stream = sh::InitializeStream<TPersistStringStream>();
    if (line)
        stream << file << ":" << line;
    else
        stream << file << ":? ";
    stream << ": ";

    sink.append(stream.str());
}

}  // namespace sh
