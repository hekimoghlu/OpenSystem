/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#ifndef WKImageCairo_h
#define WKImageCairo_h

#if USE(CAIRO)

#include <WebKit/WKBase.h>
#include <WebKit/WKImage.h>

typedef struct _cairo_surface cairo_surface_t;

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT cairo_surface_t* WKImageCreateCairoSurface(WKImageRef image);

WK_EXPORT WKImageRef WKImageCreateFromCairoSurface(cairo_surface_t* surface, WKImageOptions options);

#ifdef __cplusplus
}
#endif

#endif // USE(CAIRO)

#endif /* WKImageCairo_h */
