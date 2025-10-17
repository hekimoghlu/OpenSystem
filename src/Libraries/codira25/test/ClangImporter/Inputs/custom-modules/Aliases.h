/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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

#if defined(UNICODE)
#define F FW
#define V VW
#else
#define F FA
#define V VA
#endif

#if defined(_WIN32)
#define ALIASES_ABI /**/
#else
#define ALIASES_ABI __attribute__((__visibility__("default")))
#endif

extern ALIASES_ABI const unsigned int VA;
extern ALIASES_ABI const unsigned long long VW;

ALIASES_ABI void FA(unsigned int);
ALIASES_ABI void FW(unsigned long long);

#define InvalidCall DoesNotExist

extern ALIASES_ABI float UIA;
extern ALIASES_ABI double UIW;

#if defined(UNICODE)
#define UI UIW
#else
#define UI UIA
#endif

enum {
  ALPHA = 0,
#define ALPHA ALPHA
  BETA = 1,
#define BETA BETA
};

enum {
  _CLOCK_MONOTONIC __attribute__((__language_name__("CLOCK_MONOTONIC"))),
#define CLOCK_MONOTONIC _CLOCK_MONOTONIC
} _clock_t;

enum : int {
  overloaded,
};
#define overload overloaded
extern const int const_overloaded __attribute__((__language_name__("overload")));

void variadic(int count, ...);
#define aliased_variadic variadic
