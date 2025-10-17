/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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
#include "config.h"
#include "Icon.h"

#include "GraphicsContext.h"
#include "LocalWindowsContext.h"
#include <windows.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

static const int shell32MultipleFileIconIndex = 54;

Icon::Icon(HICON icon)
    : m_hIcon(icon)
{
    ASSERT(icon);
}

Icon::~Icon()
{
    DestroyIcon(m_hIcon);
}

// FIXME: Move the code to ChromeClient::iconForFiles().
RefPtr<Icon> Icon::createIconForFiles(const Vector<String>& filenames)
{
    if (filenames.isEmpty())
        return nullptr;

    if (filenames.size() == 1) {
        SHFILEINFO sfi;
        memset(&sfi, 0, sizeof(sfi));

        String tmpFilename = filenames[0];
        if (!SHGetFileInfo(tmpFilename.wideCharacters().data(), 0, &sfi, sizeof(sfi), SHGFI_ICON | SHGFI_SHELLICONSIZE | SHGFI_SMALLICON))
            return nullptr;

        return adoptRef(new Icon(sfi.hIcon));
    }

    WCHAR buffer[MAX_PATH];
    UINT length = ::GetSystemDirectoryW(buffer, std::size(buffer));
    if (!length)
        return nullptr;

    if (wcscat_s(buffer, L"\\shell32.dll"))
        return nullptr;

    HICON hIcon;
    if (!::ExtractIconExW(buffer, shell32MultipleFileIconIndex, 0, &hIcon, 1))
        return nullptr;
    return adoptRef(new Icon(hIcon));
}

void Icon::paint(GraphicsContext& context, const FloatRect& r)
{
    if (context.paintingDisabled())
        return;

    LocalWindowsContext windowContext(context, enclosingIntRect(r));
    DrawIconEx(windowContext.hdc(), r.x(), r.y(), m_hIcon, r.width(), r.height(), 0, 0, DI_NORMAL);
}

}
