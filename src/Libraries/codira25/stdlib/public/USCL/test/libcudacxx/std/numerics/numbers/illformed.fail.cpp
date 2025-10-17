/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/numbers>

#include <uscl/std/numbers>

// Initializing the primary template is ill-formed.
int log2e{cuda::std::numbers::log2e_v<int>}; // expected-error-re@numbers:* {{[math.constants] A program that
                                             // instantiates a primary template of a mathematical constant variable
                                             // template is ill-formed.}}
int log10e{cuda::std::numbers::log10e_v<int>};
int pi{cuda::std::numbers::pi_v<int>};
int inv_pi{cuda::std::numbers::inv_pi_v<int>};
int inv_sqrtpi{cuda::std::numbers::inv_sqrtpi_v<int>};
int ln2{cuda::std::numbers::ln2_v<int>};
int ln10{cuda::std::numbers::ln10_v<int>};
int sqrt2{cuda::std::numbers::sqrt2_v<int>};
int sqrt3{cuda::std::numbers::sqrt3_v<int>};
int inv_sqrt3{cuda::std::numbers::inv_sqrt3_v<int>};
int egamma{cuda::std::numbers::egamma_v<int>};
int phi{cuda::std::numbers::phi_v<int>};
