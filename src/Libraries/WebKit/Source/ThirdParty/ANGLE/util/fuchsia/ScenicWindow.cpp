/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
// ScenicWindow.cpp:
//    Implements methods from ScenicWindow
//

#include "util/fuchsia/ScenicWindow.h"

#include <fuchsia/element/cpp/fidl.h>
#include <lib/async-loop/cpp/loop.h>
#include <lib/async-loop/default.h>
#include <lib/fdio/directory.h>
#include <lib/fidl/cpp/interface_ptr.h>
#include <lib/fidl/cpp/interface_request.h>
#include <lib/zx/channel.h>
#include <zircon/status.h>

namespace
{

async::Loop *GetDefaultLoop()
{
    static async::Loop *defaultLoop = new async::Loop(&kAsyncLoopConfigNeverAttachToThread);
    return defaultLoop;
}

zx::channel ConnectToServiceRoot()
{
    zx::channel clientChannel;
    zx::channel serverChannel;
    zx_status_t result = zx::channel::create(0, &clientChannel, &serverChannel);
    ASSERT(result == ZX_OK);
    result = fdio_service_connect("/svc/.", serverChannel.release());
    ASSERT(result == ZX_OK);
    return clientChannel;
}

template <typename Interface>
zx_status_t ConnectToService(zx_handle_t serviceRoot, fidl::InterfaceRequest<Interface> request)
{
    ASSERT(request.is_valid());
    return fdio_service_connect_at(serviceRoot, Interface::Name_, request.TakeChannel().release());
}

template <typename Interface>
fidl::InterfacePtr<Interface> ConnectToService(zx_handle_t serviceRoot,
                                               async_dispatcher_t *dispatcher)
{
    fidl::InterfacePtr<Interface> result;
    ConnectToService(serviceRoot, result.NewRequest(dispatcher));
    return result;
}

}  // namespace

// TODO: http://anglebug.com/42050005 - Implement using fuchsia.element.GraphicalPresenter to pass a
// ViewCreationToken to Fuchsia Flatland.
ScenicWindow::ScenicWindow()
    : mLoop(GetDefaultLoop()),
      mServiceRoot(ConnectToServiceRoot()),
      mPresenter(ConnectToService<fuchsia::element::GraphicalPresenter>(mServiceRoot.get(),
                                                                        mLoop->dispatcher()))
{}

ScenicWindow::~ScenicWindow()
{
    destroy();
}

bool ScenicWindow::initializeImpl(const std::string &name, int width, int height)
{
    return true;
}

void ScenicWindow::disableErrorMessageDialog() {}

void ScenicWindow::destroy()
{
    mFuchsiaEGLWindow.reset();
}

void ScenicWindow::resetNativeWindow()
{
    UNIMPLEMENTED();
}

EGLNativeWindowType ScenicWindow::getNativeWindow() const
{
    return reinterpret_cast<EGLNativeWindowType>(mFuchsiaEGLWindow.get());
}

EGLNativeDisplayType ScenicWindow::getNativeDisplay() const
{
    return EGL_DEFAULT_DISPLAY;
}

void ScenicWindow::messageLoop()
{
    mLoop->ResetQuit();
    mLoop->RunUntilIdle();
}

void ScenicWindow::setMousePosition(int x, int y)
{
    UNIMPLEMENTED();
}

bool ScenicWindow::setOrientation(int width, int height)
{
    UNIMPLEMENTED();
    return false;
}

bool ScenicWindow::setPosition(int x, int y)
{
    UNIMPLEMENTED();
    return false;
}

bool ScenicWindow::resize(int width, int height)
{
    fuchsia_egl_window_resize(mFuchsiaEGLWindow.get(), width, height);
    return true;
}

void ScenicWindow::setVisible(bool isVisible) {}

void ScenicWindow::signalTestEvent() {}

void ScenicWindow::present()
{
    UNIMPLEMENTED();
}

void ScenicWindow::updateViewSize()
{
    UNIMPLEMENTED();
}

// static
OSWindow *OSWindow::New()
{
    return new ScenicWindow;
}
