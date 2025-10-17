/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef LIBCUDACXX_TEST_SUPPORT_READ_WRITE_H
#define LIBCUDACXX_TEST_SUPPORT_READ_WRITE_H

struct value_type_indirection
{
  using value_type = int;
  __host__ __device__ value_type& operator*() const;
};

struct element_type_indirection
{
  using element_type = long;
  __host__ __device__ element_type& operator*() const;
};

struct proxy_indirection
{
  using value_type = int;
  __host__ __device__ value_type operator*() const;
};

struct read_only_indirection
{
  using value_type = int const;
  __host__ __device__ value_type& operator*() const;
};

// doubles as missing_iter_reference_t
struct missing_dereference
{
  using value_type = int;
};

#endif // LIBCUDACXX_TEST_SUPPORT_READ_WRITE_H
