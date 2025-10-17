/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

#ifndef __ImageMasking__
#define __ImageMasking__

#include <ApplicationServices/ApplicationServices.h>

void doOneBitMaskImages(CGContextRef context);

void doMaskImageWithMaskFromURL(CGContextRef context,
					CFURLRef imageURL, size_t imagewidth,
				    size_t imageheight, size_t bitsPerComponent, 
				    CFURLRef theMaskingImageURL, size_t maskwidth,
				    size_t maskheight);

void doMaskImageWithColorFromURL(CGContextRef context, CFURLRef url,
 					size_t width, size_t height,
					Boolean isColor);

void exportImageWithMaskFromURLWithDestination(CGContextRef context,
				    CFURLRef imageURL, size_t imagewidth,
				    size_t imageheight, size_t bitsPerComponent, 
				    CFURLRef theMaskingImageURL, size_t maskwidth,
				    size_t maskheight);

void doMaskImageWithGrayImageFromURL(CGContextRef context, CFURLRef imageURL, size_t imagewidth,
				    size_t imageheight, size_t bitsPerComponent, 
				    CFURLRef theMaskingImageURL, size_t maskwidth,
				    size_t maskheight);

void drawWithClippingMask(CGContextRef context, 
				CFURLRef theMaskingImageURL,  size_t maskwidth,
				size_t maskheight);


#endif	// __ImageMasking__
