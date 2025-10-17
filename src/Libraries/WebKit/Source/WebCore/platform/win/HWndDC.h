/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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
#ifndef HWndDC_h
#define HWndDC_h

#include <windows.h>
#include <wtf/Noncopyable.h>

namespace WebCore {

class HWndDC {
    WTF_MAKE_NONCOPYABLE(HWndDC);
public:
    HWndDC()
        : m_hwnd(0)
        , m_hdc(0)
    {
    }

    explicit HWndDC(HWND hwnd)
        : m_hwnd(hwnd)
        , m_hdc(::GetDC(hwnd))
    {
    }

    HWndDC(HWND hwnd, HRGN hrgnClip, DWORD flags)
        : m_hwnd(hwnd)
        , m_hdc(::GetDCEx(hwnd, hrgnClip, flags))
    {
    }

    ~HWndDC()
    {
        clear();
    }

    HDC setHWnd(HWND hwnd)
    {
        clear();
        m_hwnd = hwnd;
        m_hdc = ::GetDC(hwnd);
        return m_hdc;
    }

    void clear()
    {
        if (!m_hdc)
            return;
        ::ReleaseDC(m_hwnd, m_hdc);
        m_hwnd = 0;
        m_hdc = 0;
    }

    operator HDC()
    {
        return m_hdc;
    }

private:
    HWND m_hwnd;
    HDC m_hdc;
};

} // namespace WebCore

#endif // HWndDC_h
