/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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
#define STRFrect "(%d,%d-%d,%d)"
#define ARGSrect(r) (r).start_row, (r).start_col, (r).end_row, (r).end_col

/* Expand dst to contain src as well */
static void rect_expand(VTermRect *dst, VTermRect *src)
{
  if(dst->start_row > src->start_row) dst->start_row = src->start_row;
  if(dst->start_col > src->start_col) dst->start_col = src->start_col;
  if(dst->end_row   < src->end_row)   dst->end_row   = src->end_row;
  if(dst->end_col   < src->end_col)   dst->end_col   = src->end_col;
}

/* Clip the dst to ensure it does not step outside of bounds */
static void rect_clip(VTermRect *dst, VTermRect *bounds)
{
  if(dst->start_row < bounds->start_row) dst->start_row = bounds->start_row;
  if(dst->start_col < bounds->start_col) dst->start_col = bounds->start_col;
  if(dst->end_row   > bounds->end_row)   dst->end_row   = bounds->end_row;
  if(dst->end_col   > bounds->end_col)   dst->end_col   = bounds->end_col;
  /* Ensure it doesn't end up negatively-sized */
  if(dst->end_row < dst->start_row) dst->end_row = dst->start_row;
  if(dst->end_col < dst->start_col) dst->end_col = dst->start_col;
}

/* True if the two rectangles are equal */
static int rect_equal(VTermRect *a, VTermRect *b)
{
  return (a->start_row == b->start_row) &&
         (a->start_col == b->start_col) &&
         (a->end_row   == b->end_row)   &&
         (a->end_col   == b->end_col);
}

/* True if small is contained entirely within big */
static int rect_contains(VTermRect *big, VTermRect *small)
{
  if(small->start_row < big->start_row) return 0;
  if(small->start_col < big->start_col) return 0;
  if(small->end_row   > big->end_row)   return 0;
  if(small->end_col   > big->end_col)   return 0;
  return 1;
}

/* True if the rectangles overlap at all */
static int rect_intersects(VTermRect *a, VTermRect *b)
{
  if(a->start_row > b->end_row || b->start_row > a->end_row)
    return 0;
  if(a->start_col > b->end_col || b->start_col > a->end_col)
    return 0;
  return 1;
}
