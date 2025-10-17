/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
#include <optional>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/glib/GUniquePtr.h>

namespace WPE {

class CursorTheme {
    WTF_MAKE_TZONE_ALLOCATED(CursorTheme);
public:
    static std::unique_ptr<CursorTheme> create(const char* path, uint32_t size);
    static std::unique_ptr<CursorTheme> create();

    CursorTheme(GUniquePtr<char>&&, uint32_t, Vector<GUniquePtr<char>>&&);
    ~CursorTheme() = default;

    uint32_t size() const { return m_size; }

    struct CursorImage {
        uint32_t width { 0 };
        uint32_t height { 0 };
        uint32_t hotspotX { 0 };
        uint32_t hotspotY { 0 };
        uint32_t delay { 0 };
        Vector<uint32_t> pixels;
    };
    Vector<CursorImage> loadCursor(const char*, uint32_t size, std::optional<uint32_t> maxImages = std::nullopt);

private:
    GUniquePtr<char> m_path;
    uint32_t m_size { 0 };
    Vector<GUniquePtr<char>> m_inherited;
};

} // namespace WPE
