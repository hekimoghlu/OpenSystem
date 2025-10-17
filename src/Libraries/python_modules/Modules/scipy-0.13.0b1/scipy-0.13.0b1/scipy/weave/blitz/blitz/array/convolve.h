/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
#ifndef BZ_ARRAY_CONVOLVE_H
#define BZ_ARRAY_CONVOLVE_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/convolve.h> must be included after <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

template<typename T>
Array<T,1> convolve(const Array<T,1>& B, const Array<T,1>& C);

BZ_NAMESPACE_END

#include <blitz/array/convolve.cc>

#endif // BZ_ARRAY_CONVOLVE_H
