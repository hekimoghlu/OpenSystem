/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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

//===-- lib/Evaluate/complex.cpp ------------------------------------------===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// Author: Tunjay Akbarli
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

#include "language/Compability/Evaluate/complex.h"
#include "toolchain/Support/raw_ostream.h"

namespace language::Compability::evaluate::value {

template <typename R>
ValueWithRealFlags<Complex<R>> Complex<R>::Add(
    const Complex &that, Rounding rounding) const {
  RealFlags flags;
  Part reSum{re_.Add(that.re_, rounding).AccumulateFlags(flags)};
  Part imSum{im_.Add(that.im_, rounding).AccumulateFlags(flags)};
  return {Complex{reSum, imSum}, flags};
}

template <typename R>
ValueWithRealFlags<Complex<R>> Complex<R>::Subtract(
    const Complex &that, Rounding rounding) const {
  RealFlags flags;
  Part reDiff{re_.Subtract(that.re_, rounding).AccumulateFlags(flags)};
  Part imDiff{im_.Subtract(that.im_, rounding).AccumulateFlags(flags)};
  return {Complex{reDiff, imDiff}, flags};
}

template <typename R>
ValueWithRealFlags<Complex<R>> Complex<R>::Multiply(
    const Complex &that, Rounding rounding) const {
  // (a + ib)*(c + id) -> ac - bd + i(ad + bc)
  RealFlags flags;
  Part ac{re_.Multiply(that.re_, rounding).AccumulateFlags(flags)};
  Part bd{im_.Multiply(that.im_, rounding).AccumulateFlags(flags)};
  Part ad{re_.Multiply(that.im_, rounding).AccumulateFlags(flags)};
  Part bc{im_.Multiply(that.re_, rounding).AccumulateFlags(flags)};
  Part acbd{ac.Subtract(bd, rounding).AccumulateFlags(flags)};
  Part adbc{ad.Add(bc, rounding).AccumulateFlags(flags)};
  return {Complex{acbd, adbc}, flags};
}

template <typename R>
ValueWithRealFlags<Complex<R>> Complex<R>::Divide(
    const Complex &that, Rounding rounding) const {
  // (a + ib)/(c + id) -> [(a+ib)*(c-id)] / [(c+id)*(c-id)]
  //   -> [ac+bd+i(bc-ad)] / (cc+dd)  -- note (cc+dd) is real
  //   -> ((ac+bd)/(cc+dd)) + i((bc-ad)/(cc+dd))
  RealFlags flags;
  Part cc{that.re_.Multiply(that.re_, rounding).AccumulateFlags(flags)};
  Part dd{that.im_.Multiply(that.im_, rounding).AccumulateFlags(flags)};
  Part ccPdd{cc.Add(dd, rounding).AccumulateFlags(flags)};
  if (!flags.test(RealFlag::Overflow) && !flags.test(RealFlag::Underflow)) {
    // den = (cc+dd) did not overflow or underflow; try the naive
    // sequence without scaling to avoid extra roundings.
    Part ac{re_.Multiply(that.re_, rounding).AccumulateFlags(flags)};
    Part ad{re_.Multiply(that.im_, rounding).AccumulateFlags(flags)};
    Part bc{im_.Multiply(that.re_, rounding).AccumulateFlags(flags)};
    Part bd{im_.Multiply(that.im_, rounding).AccumulateFlags(flags)};
    Part acPbd{ac.Add(bd, rounding).AccumulateFlags(flags)};
    Part bcSad{bc.Subtract(ad, rounding).AccumulateFlags(flags)};
    Part re{acPbd.Divide(ccPdd, rounding).AccumulateFlags(flags)};
    Part im{bcSad.Divide(ccPdd, rounding).AccumulateFlags(flags)};
    if (!flags.test(RealFlag::Overflow) && !flags.test(RealFlag::Underflow)) {
      return {Complex{re, im}, flags};
    }
  }
  // Scale numerator and denominator by d/c (if c>=d) or c/d (if c<d)
  flags.clear();
  Part scale; // will be <= 1.0 in magnitude
  bool cGEd{that.re_.ABS().Compare(that.im_.ABS()) != Relation::Less};
  if (cGEd) {
    scale = that.im_.Divide(that.re_, rounding).AccumulateFlags(flags);
  } else {
    scale = that.re_.Divide(that.im_, rounding).AccumulateFlags(flags);
  }
  Part den;
  if (cGEd) {
    Part dS{scale.Multiply(that.im_, rounding).AccumulateFlags(flags)};
    den = dS.Add(that.re_, rounding).AccumulateFlags(flags);
  } else {
    Part cS{scale.Multiply(that.re_, rounding).AccumulateFlags(flags)};
    den = cS.Add(that.im_, rounding).AccumulateFlags(flags);
  }
  Part aS{scale.Multiply(re_, rounding).AccumulateFlags(flags)};
  Part bS{scale.Multiply(im_, rounding).AccumulateFlags(flags)};
  Part re1, im1;
  if (cGEd) {
    re1 = re_.Add(bS, rounding).AccumulateFlags(flags);
    im1 = im_.Subtract(aS, rounding).AccumulateFlags(flags);
  } else {
    re1 = aS.Add(im_, rounding).AccumulateFlags(flags);
    im1 = bS.Subtract(re_, rounding).AccumulateFlags(flags);
  }
  Part re{re1.Divide(den, rounding).AccumulateFlags(flags)};
  Part im{im1.Divide(den, rounding).AccumulateFlags(flags)};
  return {Complex{re, im}, flags};
}

template <typename R> std::string Complex<R>::DumpHexadecimal() const {
  std::string result{'('};
  result += re_.DumpHexadecimal();
  result += ',';
  result += im_.DumpHexadecimal();
  result += ')';
  return result;
}

template <typename R>
toolchain::raw_ostream &Complex<R>::AsFortran(toolchain::raw_ostream &o, int kind) const {
  re_.AsFortran(o << '(', kind);
  im_.AsFortran(o << ',', kind);
  return o << ')';
}

template class Complex<Real<Integer<16>, 11>>;
template class Complex<Real<Integer<16>, 8>>;
template class Complex<Real<Integer<32>, 24>>;
template class Complex<Real<Integer<64>, 53>>;
template class Complex<Real<X87IntegerContainer, 64>>;
template class Complex<Real<Integer<128>, 113>>;
} // namespace language::Compability::evaluate::value
