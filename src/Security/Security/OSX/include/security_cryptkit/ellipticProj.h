/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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
#ifndef	_CRYPTKIT_ELLIPTIC_PROJ_H_
#define _CRYPTKIT_ELLIPTIC_PROJ_H_

#include "ckconfig.h"

#include "giantIntegers.h"
#include "curveParams.h"

/*
 * A projective point.
 */
typedef struct {
	giant x;
	giant y;
	giant z;
} pointProjStruct;

typedef pointProjStruct *pointProj;

pointProj  /* Allocates a new projective point. */
newPointProj(unsigned numDigits);

void  /* Frees point. */
freePointProj(pointProj pt);

void  /* Copies point to point; pt2 := pt1. */
ptopProj(pointProj pt1, pointProj pt2);

void /* Point doubling. */
ellDoubleProj(pointProj pt, curveParams *cp);

void /* Point adding; pt0 := pt0 - pt1. */
ellAddProj(pointProj pt0, pointProj pt1, curveParams *cp);

void /* Point negation; pt := -pt. */
ellNegProj(pointProj pt, curveParams *cp);

void /* Point subtraction; pt0 := pt0 - pt1. */
ellSubProj(pointProj pt0, pointProj pt1, curveParams *cp);

void /* pt := pt * k, result normalized */
ellMulProjSimple(pointProj pt0, giant k, curveParams *cp);

void /* General elliptic mul; pt1 := k*pt0. */
ellMulProj(pointProj pt0, pointProj pt1, giant k, curveParams *cp);

void /* Generate normalized point (X, Y, 1) from given (x,y,z). */
normalizeProj(pointProj pt, curveParams *cp);

void /* Find a point (x, y, 1) on the curve. */
findPointProj(pointProj pt, giant seed, curveParams *cp);

#endif	/* _CRYPTKIT_ELLIPTIC_PROJ_H_ */
