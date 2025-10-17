/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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

#include <optional>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/CString.h>

namespace WPE {

class CursorTheme;

namespace DRM {

class CursorTheme {
    WTF_MAKE_TZONE_ALLOCATED(CursorTheme);
public:
    static std::unique_ptr<CursorTheme> create(const char* path, uint32_t size);
    static std::unique_ptr<CursorTheme> create();

    CursorTheme(std::unique_ptr<WPE::CursorTheme>&&);
    ~CursorTheme();

    struct Image {
        uint32_t width { 0 };
        uint32_t height { 0 };
        uint32_t hotspotX { 0 };
        uint32_t hotspotY { 0 };
        Vector<uint32_t> pixels;
    };
    const Vector<Image>& cursor(const char*, double, std::optional<uint32_t> maxImages = std::nullopt);

private:
    void loadCursor(const char*, double, std::optional<uint32_t> maxImages, Vector<CursorTheme::Image>&);

    std::unique_ptr<WPE::CursorTheme> m_theme;
    HashMap<std::pair<CString, uint32_t>, Vector<Image>> m_cursors;
};

} // namespace DRM

} // namespace WPE
