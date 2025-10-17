/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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
#pragma once

#include "CSSCalcSymbolTable.h"
#include "CSSPrimitiveNumericRange.h"
#include "CSSToLengthConversionData.h"

namespace WebCore {

namespace Calculation {
enum class Category : uint8_t;
}

namespace CSSCalc {

struct Child;
struct Tree;

struct Abs;
struct Acos;
struct Anchor;
struct AnchorSize;
struct Asin;
struct Atan2;
struct Atan;
struct CanonicalDimension;
struct Clamp;
struct ContainerProgress;
struct Cos;
struct Exp;
struct Hypot;
struct Invert;
struct Log;
struct Max;
struct MediaProgress;
struct Min;
struct Mod;
struct Negate;
struct NonCanonicalDimension;
struct Number;
struct Percentage;
struct Pow;
struct Product;
struct Progress;
struct Random;
struct Rem;
struct RoundDown;
struct RoundNearest;
struct RoundToZero;
struct RoundUp;
struct Sign;
struct Sin;
struct Sqrt;
struct Sum;
struct Symbol;
struct Tan;

// https://drafts.csswg.org/css-values-4/#calc-simplification

struct SimplificationOptions {
    // `category` represents the context in which the simplification is taking place.
    Calculation::Category category;

    // `range` represents the allowed numeric range for the calculated result.
    CSS::Range range;

    // `conversionData` contains information needed to convert length units into their canonical forms.
    std::optional<CSSToLengthConversionData> conversionData;

    // `symbolTable` contains information needed to convert unresolved symbols into Numeric values.
    CSSCalcSymbolTable symbolTable;

    // `allowZeroValueLengthRemovalFromSum` allows removal of 0 value lengths (px, em, etc.) from Sum operations.
    bool allowZeroValueLengthRemovalFromSum = false;
};

// MARK: Copy & Simplify

Tree copyAndSimplify(const Tree&, const SimplificationOptions&);
Child copyAndSimplify(const Child&, const SimplificationOptions&);

// MARK: In-place Simplify

std::optional<Child> simplify(Number&, const SimplificationOptions&);
std::optional<Child> simplify(Percentage&, const SimplificationOptions&);
std::optional<Child> simplify(NonCanonicalDimension&, const SimplificationOptions&);
std::optional<Child> simplify(CanonicalDimension&, const SimplificationOptions&);
std::optional<Child> simplify(Symbol&, const SimplificationOptions&);
std::optional<Child> simplify(Sum&, const SimplificationOptions&);
std::optional<Child> simplify(Product&, const SimplificationOptions&);
std::optional<Child> simplify(Negate&, const SimplificationOptions&);
std::optional<Child> simplify(Invert&, const SimplificationOptions&);
std::optional<Child> simplify(Min&, const SimplificationOptions&);
std::optional<Child> simplify(Max&, const SimplificationOptions&);
std::optional<Child> simplify(Clamp&, const SimplificationOptions&);
std::optional<Child> simplify(RoundNearest&, const SimplificationOptions&);
std::optional<Child> simplify(RoundUp&, const SimplificationOptions&);
std::optional<Child> simplify(RoundDown&, const SimplificationOptions&);
std::optional<Child> simplify(RoundToZero&, const SimplificationOptions&);
std::optional<Child> simplify(Mod&, const SimplificationOptions&);
std::optional<Child> simplify(Rem&, const SimplificationOptions&);
std::optional<Child> simplify(Sin&, const SimplificationOptions&);
std::optional<Child> simplify(Cos&, const SimplificationOptions&);
std::optional<Child> simplify(Tan&, const SimplificationOptions&);
std::optional<Child> simplify(Asin&, const SimplificationOptions&);
std::optional<Child> simplify(Acos&, const SimplificationOptions&);
std::optional<Child> simplify(Atan&, const SimplificationOptions&);
std::optional<Child> simplify(Atan2&, const SimplificationOptions&);
std::optional<Child> simplify(Pow&, const SimplificationOptions&);
std::optional<Child> simplify(Sqrt&, const SimplificationOptions&);
std::optional<Child> simplify(Hypot&, const SimplificationOptions&);
std::optional<Child> simplify(Log&, const SimplificationOptions&);
std::optional<Child> simplify(Exp&, const SimplificationOptions&);
std::optional<Child> simplify(Abs&, const SimplificationOptions&);
std::optional<Child> simplify(Sign&, const SimplificationOptions&);
std::optional<Child> simplify(Random&, const SimplificationOptions&);
std::optional<Child> simplify(Progress&, const SimplificationOptions&);
std::optional<Child> simplify(MediaProgress&, const SimplificationOptions&);
std::optional<Child> simplify(ContainerProgress&, const SimplificationOptions&);
std::optional<Child> simplify(Anchor&, const SimplificationOptions&);
std::optional<Child> simplify(AnchorSize&, const SimplificationOptions&);

// MARK: Unit Canonicalization

std::optional<CanonicalDimension> canonicalize(NonCanonicalDimension, const std::optional<CSSToLengthConversionData>&);

} // namespace CSSCalc
} // namespace WebCore
