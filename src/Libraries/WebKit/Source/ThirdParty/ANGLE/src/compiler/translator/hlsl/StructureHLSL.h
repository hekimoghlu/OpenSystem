/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// StructureHLSL.h:
//   HLSL translation of GLSL constructors and structures.
//

#ifndef COMPILER_TRANSLATOR_HLSL_STRUCTUREHLSL_H_
#define COMPILER_TRANSLATOR_HLSL_STRUCTUREHLSL_H_

#include "compiler/translator/Common.h"
#include "compiler/translator/IntermNode.h"

#include <set>

class TInfoSinkBase;
class TScopeBracket;

namespace sh
{

// This helper class assists structure and interface block definitions in determining
// how to pack std140 structs within HLSL's packing rules.
class Std140PaddingHelper
{
  public:
    explicit Std140PaddingHelper(const std::map<TString, int> &structElementIndexes,
                                 unsigned int *uniqueCounter);
    Std140PaddingHelper(const Std140PaddingHelper &other);
    Std140PaddingHelper &operator=(const Std140PaddingHelper &other);

    int elementIndex() const { return mElementIndex; }
    int prePadding(const TType &type, bool forcePadding);
    TString prePaddingString(const TType &type, bool forcePadding);
    TString postPaddingString(const TType &type,
                              bool useHLSLRowMajorPacking,
                              bool isLastElement,
                              bool forcePadding);

  private:
    TString next();

    unsigned *mPaddingCounter;
    int mElementIndex;
    const std::map<TString, int> *mStructElementIndexes;
};

class StructureHLSL : angle::NonCopyable
{
  public:
    StructureHLSL();

    // Returns the name of the constructor function.
    TString addStructConstructor(const TStructure &structure);
    TString addBuiltInConstructor(const TType &type, const TIntermSequence *parameters);

    static TString defineNameless(const TStructure &structure);
    void ensureStructDefined(const TStructure &structure);

    std::string structsHeader() const;

    Std140PaddingHelper getPaddingHelper();

  private:
    unsigned mUniquePaddingCounter;

    std::map<TString, int> mStd140StructElementIndexes;

    struct TStructProperties : public angle::NonCopyable
    {
        POOL_ALLOCATOR_NEW_DELETE

        TStructProperties() {}

        // Constructor is an empty string in case the struct doesn't have a constructor yet.
        TString constructor;
    };

    // Map from struct name to struct properties.
    typedef std::map<TString, TStructProperties *> DefinedStructs;
    DefinedStructs mDefinedStructs;

    // Struct declarations need to be kept in a vector instead of having them inside mDefinedStructs
    // since maintaining the original order is necessary for nested structs.
    typedef std::vector<TString> StructDeclarations;
    StructDeclarations mStructDeclarations;

    typedef std::set<TString> BuiltInConstructors;
    BuiltInConstructors mBuiltInConstructors;

    void storeStd140ElementIndex(const TStructure &structure, bool useHLSLRowMajorPacking);
    TString defineQualified(const TStructure &structure,
                            bool useHLSLRowMajorPacking,
                            bool useStd140Packing,
                            bool forcePackingEnd);
    DefinedStructs::iterator defineVariants(const TStructure &structure, const TString &name);
};
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_HLSL_STRUCTUREHLSL_H_
