/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// MultiThreadSteps.cpp:
//   Synchronization help for tests that use multiple threads.

#include "MultiThreadSteps.h"

#include "angle_test_platform.h"
#include "gtest/gtest.h"
#include "util/EGLWindow.h"

namespace angle
{

void RunLockStepThreads(EGLWindow *window, size_t threadCount, LockStepThreadFunc threadFuncs[])
{
    constexpr EGLint kPBufferSize = 256;
    RunLockStepThreadsWithSize(window, kPBufferSize, kPBufferSize, threadCount, threadFuncs);
}

void RunLockStepThreadsWithSize(EGLWindow *window,
                                EGLint width,
                                EGLint height,
                                size_t threadCount,
                                LockStepThreadFunc threadFuncs[])
{
    EGLDisplay dpy   = window->getDisplay();
    EGLConfig config = window->getConfig();

    // Initialize the pbuffer and context
    EGLint pbufferAttributes[] = {
        EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE, EGL_NONE,
    };

    std::vector<EGLSurface> surfaces(threadCount);
    std::vector<EGLContext> contexts(threadCount);

    // Create N surfaces and shared contexts, one for each thread
    for (size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex)
    {
        surfaces[threadIndex] = eglCreatePbufferSurface(dpy, config, pbufferAttributes);
        EXPECT_EQ(eglGetError(), EGL_SUCCESS);
        EGLint extraAttributes[] = {EGL_CONTEXT_VIRTUALIZATION_GROUP_ANGLE,
                                    static_cast<EGLint>(threadIndex), EGL_NONE};
        if (!IsEGLDisplayExtensionEnabled(dpy, "EGL_ANGLE_context_virtualization"))
        {
            extraAttributes[0] = EGL_NONE;
        }
        contexts[threadIndex] =
            window->createContext(threadIndex == 0 ? EGL_NO_CONTEXT : contexts[0], extraAttributes);
        EXPECT_NE(EGL_NO_CONTEXT, contexts[threadIndex]) << threadIndex;
    }

    std::vector<std::thread> threads(threadCount);

    // Run the threads
    for (size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex)
    {
        threads[threadIndex] = std::thread(std::move(threadFuncs[threadIndex]), dpy,
                                           surfaces[threadIndex], contexts[threadIndex]);
    }

    // Wait for them to finish
    for (size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex)
    {
        threads[threadIndex].join();

        // Clean up
        eglDestroySurface(dpy, surfaces[threadIndex]);
        eglDestroyContext(dpy, contexts[threadIndex]);
    }
}
}  // namespace angle
