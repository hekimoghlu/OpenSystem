/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
#include "RefPtrCairo.h"

#if USE(CAIRO) || PLATFORM(GTK)

#include <cairo.h>

namespace WTF {

cairo_t* DefaultRefDerefTraits<cairo_t>::refIfNotNull(cairo_t* ptr)
{
    if (LIKELY(ptr))
        cairo_reference(ptr);
    return ptr;
}

void DefaultRefDerefTraits<cairo_t>::derefIfNotNull(cairo_t* ptr)
{
    if (LIKELY(ptr))
        cairo_destroy(ptr);
}

cairo_surface_t* DefaultRefDerefTraits<cairo_surface_t>::refIfNotNull(cairo_surface_t* ptr)
{
    if (LIKELY(ptr))
        cairo_surface_reference(ptr);
    return ptr;
}

void DefaultRefDerefTraits<cairo_surface_t>::derefIfNotNull(cairo_surface_t* ptr)
{
    if (LIKELY(ptr))
        cairo_surface_destroy(ptr);
}

cairo_font_face_t* DefaultRefDerefTraits<cairo_font_face_t>::refIfNotNull(cairo_font_face_t* ptr)
{
    if (LIKELY(ptr))
        cairo_font_face_reference(ptr);
    return ptr;
}

void DefaultRefDerefTraits<cairo_font_face_t>::derefIfNotNull(cairo_font_face_t* ptr)
{
    if (LIKELY(ptr))
        cairo_font_face_destroy(ptr);
}

cairo_scaled_font_t* DefaultRefDerefTraits<cairo_scaled_font_t>::refIfNotNull(cairo_scaled_font_t* ptr)
{
    if (LIKELY(ptr))
        cairo_scaled_font_reference(ptr);
    return ptr;
}

void DefaultRefDerefTraits<cairo_scaled_font_t>::derefIfNotNull(cairo_scaled_font_t* ptr)
{
    if (LIKELY(ptr))
        cairo_scaled_font_destroy(ptr);
}

cairo_pattern_t* DefaultRefDerefTraits<cairo_pattern_t>::refIfNotNull(cairo_pattern_t* ptr)
{
    if (LIKELY(ptr))
        cairo_pattern_reference(ptr);
    return ptr;
}

void DefaultRefDerefTraits<cairo_pattern_t>::derefIfNotNull(cairo_pattern_t* ptr)
{
    if (LIKELY(ptr))
        cairo_pattern_destroy(ptr);
}

cairo_region_t* DefaultRefDerefTraits<cairo_region_t>::refIfNotNull(cairo_region_t* ptr)
{
    if (LIKELY(ptr))
        cairo_region_reference(ptr);
    return ptr;
}

void DefaultRefDerefTraits<cairo_region_t>::derefIfNotNull(cairo_region_t* ptr)
{
    if (LIKELY(ptr))
        cairo_region_destroy(ptr);
}

} // namespace WTF

#endif // USE(CAIRO) || PLATFORM(GTK)
