/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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
/* definitions */

typedef struct  /* This is how to define a projective point. */
{
	 giant x;
	 giant y; 
	 giant z;
} point_struct_proj;

typedef point_struct_proj *point_proj;

point_proj  /* Allocates a new projective point. */
new_point_proj(int shorts);

void  /* Frees point. */
free_point_proj(point_proj pt);

void  /* Copies point to point. */
ptop_proj(point_proj pt1, point_proj pt2);

void  /* Initialization. */
init_ell_proj(int shorts);

void /* Point doubling. */
ell_double_proj(point_proj pt, giant a, giant p);

void /* Point addition. */
ell_add_proj(point_proj pt0, point_proj pt1, giant a, giant p);

void /* Point negation. */
ell_neg_proj(point_proj pt, giant p);

void /* Point subtraction. */
ell_sub_proj(point_proj pt0, point_proj pt1, giant a, giant p);

void /* General elliptic mul. */
ell_mul_proj(point_proj pt0, point_proj pt1, giant k, giant a, giant p);

void /* Generate normalized point (X, Y, 1) from given (x,y,z). */
normalize_proj(point_proj pt, giant p);

void /* Find a point (x, y, 1) on the curve. */
find_point_proj(point_proj pt, giant seed, giant a, giant b, giant p);

