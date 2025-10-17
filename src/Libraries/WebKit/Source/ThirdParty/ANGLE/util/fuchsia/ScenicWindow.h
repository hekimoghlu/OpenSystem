/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ScenicWindow.h:
//    Subclasses OSWindow for Fuchsia's Scenic compositor.
//

#ifndef UTIL_FUCHSIA_SCENIC_WINDOW_H
#define UTIL_FUCHSIA_SCENIC_WINDOW_H

#include "common/debug.h"
#include "util/OSWindow.h"
#include "util/util_export.h"

// Disable ANGLE-specific warnings that pop up in fuchsia headers.
ANGLE_DISABLE_DESTRUCTOR_OVERRIDE_WARNING
ANGLE_DISABLE_SUGGEST_OVERRIDE_WARNINGS

#include <fuchsia/element/cpp/fidl.h>
#include <fuchsia_egl.h>
#include <lib/async-loop/cpp/loop.h>
#include <zircon/types.h>
#include <string>

ANGLE_REENABLE_SUGGEST_OVERRIDE_WARNINGS
ANGLE_REENABLE_DESTRUCTOR_OVERRIDE_WARNING

struct FuchsiaEGLWindowDeleter
{
    void operator()(fuchsia_egl_window *eglWindow) { fuchsia_egl_window_destroy(eglWindow); }
};

class ANGLE_UTIL_EXPORT ScenicWindow : public OSWindow
{
  public:
    ScenicWindow();
    ~ScenicWindow() override;

    // OSWindow:
    void disableErrorMessageDialog() override;
    void destroy() override;
    void resetNativeWindow() override;
    EGLNativeWindowType getNativeWindow() const override;
    EGLNativeDisplayType getNativeDisplay() const override;
    void messageLoop() override;
    void setMousePosition(int x, int y) override;
    bool setOrientation(int width, int height) override;
    bool setPosition(int x, int y) override;
    bool resize(int width, int height) override;
    void setVisible(bool isVisible) override;
    void signalTestEvent() override;

    // Presents the window to Scenic.
    //
    // We need to do this once per EGL window surface after adding the
    // surface's image pipe as a child of our window.
    void present();

  private:
    bool initializeImpl(const std::string &name, int width, int height) override;
    void updateViewSize();

    // ScenicWindow async loop.
    async::Loop *const mLoop;

    // System services.
    zx::channel mServiceRoot;
    fuchsia::element::GraphicalPresenterPtr mPresenter;

    // EGL native window.
    std::unique_ptr<fuchsia_egl_window, FuchsiaEGLWindowDeleter> mFuchsiaEGLWindow;
};

#endif  // UTIL_FUCHSIA_SCENIC_WINDOW_H
