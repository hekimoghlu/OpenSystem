/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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

#if USE(HARFBUZZ)

#include <hb.h>

namespace WebCore {

template<typename T>
struct HbPtrDeleter {
    void operator()(T* ptr) const = delete;
};

template<typename T>
using HbUniquePtr = std::unique_ptr<T, HbPtrDeleter<T>>;

template<> struct HbPtrDeleter<hb_font_t> {
    void operator()(hb_font_t* ptr) const
    {
        hb_font_destroy(ptr);
    }
};

template<> struct HbPtrDeleter<hb_buffer_t> {
    void operator()(hb_buffer_t* ptr) const
    {
        hb_buffer_destroy(ptr);
    }
};

template<> struct HbPtrDeleter<hb_face_t> {
    void operator()(hb_face_t* ptr) const
    {
        hb_face_destroy(ptr);
    }
};

template<> struct HbPtrDeleter<hb_blob_t> {
    void operator()(hb_blob_t* ptr) const
    {
        hb_blob_destroy(ptr);
    }
};

} // namespace WebCore

using WebCore::HbUniquePtr;

#endif // USE(HARFBUZZ)
