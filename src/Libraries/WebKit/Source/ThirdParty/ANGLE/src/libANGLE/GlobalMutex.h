/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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

//
// Copyright 2023 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// GlobalMutex.h: Defines Global Mutex and utilities.

#ifndef LIBANGLE_GLOBAL_MUTEX_H_
#define LIBANGLE_GLOBAL_MUTEX_H_

#include "common/angleutils.h"

namespace egl
{
namespace priv
{
class GlobalMutex;

enum class GlobalMutexChoice
{
    // All EGL entry points except EGL Sync objects
    EGL,
    // Entry points relating to EGL Sync objects
    Sync,
};

template <GlobalMutexChoice mutexChoice>
class [[nodiscard]] ScopedGlobalMutexLock final : angle::NonCopyable
{
  public:
    ScopedGlobalMutexLock();
    ~ScopedGlobalMutexLock();

#if !defined(ANGLE_ENABLE_GLOBAL_MUTEX_LOAD_TIME_ALLOCATE)
  private:
    GlobalMutex *mMutex;
#endif
};
}  // namespace priv

using ScopedGlobalEGLMutexLock = priv::ScopedGlobalMutexLock<priv::GlobalMutexChoice::EGL>;
using ScopedGlobalEGLSyncObjectMutexLock =
    priv::ScopedGlobalMutexLock<priv::GlobalMutexChoice::Sync>;

// For Context protection where lock is optional. Works slower than ScopedGlobalMutexLock.
class [[nodiscard]] ScopedOptionalGlobalMutexLock final : angle::NonCopyable
{
  public:
    explicit ScopedOptionalGlobalMutexLock(bool enabled);
    ~ScopedOptionalGlobalMutexLock();

  private:
    priv::GlobalMutex *mMutex;
};

#if defined(ANGLE_PLATFORM_WINDOWS) && !defined(ANGLE_STATIC)
void AllocateGlobalMutex();
void DeallocateGlobalMutex();
#endif

}  // namespace egl

#endif  // LIBANGLE_GLOBAL_MUTEX_H_
