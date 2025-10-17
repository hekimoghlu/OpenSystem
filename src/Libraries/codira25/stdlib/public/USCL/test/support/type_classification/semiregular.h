/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#ifndef TEST_SUPPORT_TYPE_CLASSIFICATION_H
#define TEST_SUPPORT_TYPE_CLASSIFICATION_H

#include "copyable.h"

struct no_default_ctor
{
  __host__ __device__ no_default_ctor(int);
};
struct derived_from_non_default_initializable : no_default_ctor
{};
struct has_non_default_initializable
{
  no_default_ctor x;
};

struct deleted_default_ctor
{
  deleted_default_ctor() = delete;
};
struct derived_from_deleted_default_ctor : deleted_default_ctor
{};
struct has_deleted_default_ctor
{
  deleted_default_ctor x;
};

#endif // TEST_SUPPORT_TYPE_CLASSIFICATION_H
