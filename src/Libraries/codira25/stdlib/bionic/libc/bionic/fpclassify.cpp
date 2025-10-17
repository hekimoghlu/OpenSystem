/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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
#include <math.h>

// Legacy cruft from before we had builtin implementations of the standard macros.
// No longer declared in our <math.h>.

extern "C" int __fpclassifyd(double d) {
  return fpclassify(d);
}
__strong_alias(__fpclassify, __fpclassifyd); // glibc uses __fpclassify, BSD __fpclassifyd.

extern "C" int __fpclassifyf(float f) {
  return fpclassify(f);
}

extern "C" int __isinf(double d) {
  return isinf(d);
}
__strong_alias(isinf, __isinf);

extern "C" int __isinff(float f) {
  return isinf(f);
}
__strong_alias(isinff, __isinff);

extern "C" int __isnan(double d) {
  return isnan(d);
}
__strong_alias(isnan, __isnan);

extern "C" int __isnanf(float f) {
  return isnan(f);
}
__strong_alias(isnanf, __isnanf);

extern "C" int __isfinite(double d) {
  return isfinite(d);
}
__strong_alias(isfinite, __isfinite);

extern "C" int __isfinitef(float f) {
  return isfinite(f);
}
__strong_alias(isfinitef, __isfinitef);

extern "C" int __isnormal(double d) {
  return isnormal(d);
}
__strong_alias(isnormal, __isnormal);

extern "C" int __isnormalf(float f) {
  return isnormal(f);
}
__strong_alias(isnormalf, __isnormalf);

extern "C" int __fpclassifyl(long double ld) {
  return fpclassify(ld);
}

extern "C" int __isinfl(long double ld) {
  return isinf(ld);
}

extern "C" int __isnanl(long double ld) {
  return isnan(ld);
}

extern "C" int __isfinitel(long double ld) {
  return isfinite(ld);
}

extern "C" int __isnormall(long double ld) {
  return isnormal(ld);
}

__strong_alias(isinfl, __isinfl);
__strong_alias(isnanl, __isnanl);
__strong_alias(isfinitel, __isfinitel);
__strong_alias(isnormall, __isnormall);
