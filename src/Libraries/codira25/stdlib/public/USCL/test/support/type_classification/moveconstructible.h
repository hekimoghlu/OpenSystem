/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#ifndef TEST_SUPPORT_TYPE_CLASSIFICATION_MOVECONSTRUCTIBLE_H
#define TEST_SUPPORT_TYPE_CLASSIFICATION_MOVECONSTRUCTIBLE_H

struct HasDefaultOps
{};

struct CustomMoveCtor
{
  __host__ __device__ CustomMoveCtor(CustomMoveCtor&&) noexcept;
};

struct MoveOnly
{
  MoveOnly(MoveOnly&&) noexcept            = default;
  MoveOnly& operator=(MoveOnly&&) noexcept = default;
  MoveOnly(const MoveOnly&)                = delete;
  MoveOnly& operator=(const MoveOnly&)     = default;
};

struct CustomMoveAssign
{
  __host__ __device__ CustomMoveAssign(CustomMoveAssign&&) noexcept;
  __host__ __device__ CustomMoveAssign& operator=(CustomMoveAssign&&) noexcept;
};

struct DeletedMoveCtor
{
  DeletedMoveCtor(DeletedMoveCtor&&)            = delete;
  DeletedMoveCtor& operator=(DeletedMoveCtor&&) = default;
};

struct ImplicitlyDeletedMoveCtor
{
  DeletedMoveCtor X;
};

struct DeletedMoveAssign
{
  DeletedMoveAssign& operator=(DeletedMoveAssign&&) = delete;
};

struct ImplicitlyDeletedMoveAssign
{
  DeletedMoveAssign X;
};

class MemberLvalueReference
{
public:
  __host__ __device__ MemberLvalueReference(int&);

private:
  int& X;
};

class MemberRvalueReference
{
public:
  __host__ __device__ MemberRvalueReference(int&&);

private:
  int&& X;
};

struct NonMovable
{
  NonMovable()                        = default;
  NonMovable(NonMovable&&)            = delete;
  NonMovable& operator=(NonMovable&&) = delete;
};

struct DerivedFromNonMovable : NonMovable
{};

struct HasANonMovable
{
  NonMovable X;
};

#endif // TEST_SUPPORT_TYPE_CLASSIFICATION_MOVECONSTRUCTIBLE_H
