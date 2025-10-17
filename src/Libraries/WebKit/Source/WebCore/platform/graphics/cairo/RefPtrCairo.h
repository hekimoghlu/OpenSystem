/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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

#if USE(CAIRO) || PLATFORM(GTK)

#include <wtf/RefPtr.h>

typedef struct _cairo cairo_t;
typedef struct _cairo_surface cairo_surface_t;
typedef struct _cairo_font_face cairo_font_face_t;
typedef struct _cairo_scaled_font cairo_scaled_font_t;
typedef struct _cairo_pattern cairo_pattern_t;
typedef struct _cairo_region cairo_region_t;

namespace WTF {

template<>
struct DefaultRefDerefTraits<cairo_t> {
    WEBCORE_EXPORT static cairo_t* refIfNotNull(cairo_t*);
    WEBCORE_EXPORT static void derefIfNotNull(cairo_t*);
};

template<>
struct DefaultRefDerefTraits<cairo_surface_t> {
    WEBCORE_EXPORT static cairo_surface_t* refIfNotNull(cairo_surface_t*);
    WEBCORE_EXPORT static void derefIfNotNull(cairo_surface_t*);
};

template<>
struct DefaultRefDerefTraits<cairo_font_face_t> {
    static cairo_font_face_t* refIfNotNull(cairo_font_face_t*);
    static void derefIfNotNull(cairo_font_face_t*);
};

template<>
struct DefaultRefDerefTraits<cairo_scaled_font_t> {
    static cairo_scaled_font_t* refIfNotNull(cairo_scaled_font_t*);
    WEBCORE_EXPORT static void derefIfNotNull(cairo_scaled_font_t*);
};

template<>
struct DefaultRefDerefTraits<cairo_pattern_t> {
    static cairo_pattern_t* refIfNotNull(cairo_pattern_t*);
    static void derefIfNotNull(cairo_pattern_t*);
};

template<>
struct DefaultRefDerefTraits<cairo_region_t> {
    static cairo_region_t* refIfNotNull(cairo_region_t*);
    static void derefIfNotNull(cairo_region_t*);
};

} // namespace WTF

#endif // USE(CAIRO) || PLATFORM(GTK)
