/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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
#pragma once

#if ENABLE(FULLSCREEN_API)

#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WindowsExtras.h>

namespace WebCore {

class FullScreenClient {
public:
    virtual LRESULT fullscreenClientWndProc(HWND, UINT message, WPARAM, LPARAM) = 0;
protected:
    virtual ~FullScreenClient() = default;
};

class FullScreenWindow {
    WTF_MAKE_TZONE_ALLOCATED(FullScreenWindow);
public:
    FullScreenWindow(FullScreenClient*);
    ~FullScreenWindow();

    void createWindow(HWND ownerWindow);
    
    HWND hwnd() const { return m_hwnd; }

private:
    static LRESULT __stdcall staticWndProc(HWND, UINT message, WPARAM, LPARAM);
    LRESULT wndProc(HWND, UINT message, WPARAM, LPARAM);

    FullScreenClient* m_client;
    HWND m_hwnd;
};

} // namespace WebCore

#endif // ENABLE(FULLSCREEN_API)
