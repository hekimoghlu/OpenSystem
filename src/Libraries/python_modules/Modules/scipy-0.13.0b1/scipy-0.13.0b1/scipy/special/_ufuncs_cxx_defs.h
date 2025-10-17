/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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

#ifndef UFUNCS_PROTO_H
#define UFUNCS_PROTO_H 1
#include "_faddeeva.h"
npy_double faddeeva_dawsn(npy_double);
npy_cdouble faddeeva_dawsn_complex(npy_cdouble);
npy_cdouble faddeeva_erf(npy_cdouble);
npy_cdouble faddeeva_erfc(npy_cdouble);
npy_double faddeeva_erfcx(npy_double);
npy_cdouble faddeeva_erfcx_complex(npy_cdouble);
npy_double faddeeva_erfi(npy_double);
npy_cdouble faddeeva_erfi_complex(npy_cdouble);
npy_cdouble faddeeva_w(npy_cdouble);
#endif
