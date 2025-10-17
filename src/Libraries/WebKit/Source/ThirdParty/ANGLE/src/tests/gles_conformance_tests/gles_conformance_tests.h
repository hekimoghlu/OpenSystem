/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef CONFORMANCE_TESTS_CONFORMANCE_TEST_H_
#define CONFORMANCE_TESTS_CONFORMANCE_TEST_H_

#include "gtest/gtest.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <string>

struct D3D9
{
    static EGLNativeDisplayType GetNativeDisplay() { return EGL_DEFAULT_DISPLAY; }
};

struct D3D11
{
    static EGLNativeDisplayType GetNativeDisplay() { return EGL_D3D11_ONLY_DISPLAY_ANGLE; }
};

#define CONFORMANCE_TESTS_ES2 2
#define CONFORMANCE_TESTS_ES3 3

#if CONFORMANCE_TESTS_TYPE == CONFORMANCE_TESTS_ES2
typedef testing::Types<D3D9, D3D11> ConformanceTestTypes;
#elif CONFORMANCE_TESTS_TYPE == CONFORMANCE_TESTS_ES3
typedef testing::Types<D3D11> ConformanceTestTypes;
#else
#    error "Unknown CONFORMANCE_TESTS_TYPE"
#endif

#define DEFINE_CONFORMANCE_TEST_CLASS(name) \
    template <typename T>                   \
    class name : public ConformanceTest<T>  \
    {};                                     \
    TYPED_TEST_SUITE(name, ConformanceTestTypes);

template <typename T>
class ConformanceTest : public testing::Test
{
  public:
    ConformanceTest() : mNativeDisplay(T::GetNativeDisplay()) {}

  protected:
    void run(const std::string &testPath) { RunConformanceTest(testPath, mNativeDisplay); }

  private:
    EGLNativeDisplayType mNativeDisplay;
};

void RunConformanceTest(const std::string &testPath, EGLNativeDisplayType nativeDisplay);

#endif  // CONFORMANCE_TESTS_CONFORMANCE_TEST_H_
