/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// This file is somewhat automatically generated. Disable clang-format.
// clang-format off


#ifndef __CCCL_VERSION_H
#define __CCCL_VERSION_H

#define CCCL_VERSION 3002000
#define CCCL_MAJOR_VERSION (CCCL_VERSION / 1000000)
#define CCCL_MINOR_VERSION (((CCCL_VERSION / 1000) % 1000))
#define CCCL_PATCH_VERSION (CCCL_VERSION % 1000)

#if CCCL_PATCH_VERSION > 99
#  error "CCCL patch version cannot be greater than 99 for compatibility with Thrust/CUB's MMMmmmpp format."
#endif

#endif // __CCCL_VERSION_H
