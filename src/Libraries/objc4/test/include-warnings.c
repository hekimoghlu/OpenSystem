/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
// Detect warnings inside any header.
// The build command above filters out warnings inside non-objc headers 
// (which are noisy with -Weverything).
// -Wno-undef suppresses warnings about `#if __cplusplus` and the like.
// -Wno-old-style-cast is tough to avoid in mixed C/C++ code.
// -Wno-nullability-extension disables a warning about non-portable
//   _Nullable etc which we already handle correctly in objc-abi.h.
// -Wno-c++98-compat disables warnings about things that already
//   have guards against C++98.
// -Wno-declaration-after-statement disables a warning about mixing declarations
//   and code while building for a standard earlier than C99

#include "includes.c"
