/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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

#include <cairo.h>
#include <memory>

namespace WebCore {

template<typename T> struct CairoPtrDeleter {
    void operator()(T* ptr) const = delete;
};

template<typename T>
using CairoUniquePtr = std::unique_ptr<T, CairoPtrDeleter<T>>;

template<> struct CairoPtrDeleter<cairo_font_options_t> {
    void operator() (cairo_font_options_t* ptr) const
    {
        cairo_font_options_destroy(ptr);
    }
};

template<> struct CairoPtrDeleter<cairo_path_t> {
    void operator() (cairo_path_t* ptr) const
    {
        cairo_path_destroy(ptr);
    }
};

} // namespace WebCore
