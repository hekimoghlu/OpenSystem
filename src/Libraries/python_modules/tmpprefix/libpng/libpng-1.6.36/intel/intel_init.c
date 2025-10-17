/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
#include "../pngpriv.h"

#ifdef PNG_READ_SUPPORTED
#if PNG_INTEL_SSE_IMPLEMENTATION > 0

void
png_init_filter_functions_sse2(png_structp pp, unsigned int bpp)
{
   /* The techniques used to implement each of these filters in SSE operate on
    * one pixel at a time.
    * So they generally speed up 3bpp images about 3x, 4bpp images about 4x.
    * They can scale up to 6 and 8 bpp images and down to 2 bpp images,
    * but they'd not likely have any benefit for 1bpp images.
    * Most of these can be implemented using only MMX and 64-bit registers,
    * but they end up a bit slower than using the equally-ubiquitous SSE2.
   */
   png_debug(1, "in png_init_filter_functions_sse2");
   if (bpp == 3)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub3_sse2;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg3_sse2;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] =
         png_read_filter_row_paeth3_sse2;
   }
   else if (bpp == 4)
   {
      pp->read_filter[PNG_FILTER_VALUE_SUB-1] = png_read_filter_row_sub4_sse2;
      pp->read_filter[PNG_FILTER_VALUE_AVG-1] = png_read_filter_row_avg4_sse2;
      pp->read_filter[PNG_FILTER_VALUE_PAETH-1] =
          png_read_filter_row_paeth4_sse2;
   }

   /* No need optimize PNG_FILTER_VALUE_UP.  The compiler should
    * autovectorize.
    */
}

#endif /* PNG_INTEL_SSE_IMPLEMENTATION > 0 */
#endif /* PNG_READ_SUPPORTED */
