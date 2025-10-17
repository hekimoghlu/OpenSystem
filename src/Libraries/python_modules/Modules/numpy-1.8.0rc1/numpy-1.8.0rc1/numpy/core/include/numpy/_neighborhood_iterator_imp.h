/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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

#ifndef _NPY_INCLUDE_NEIGHBORHOOD_IMP
#error You should not include this header directly
#endif
/*
 * Private API (here for inline)
 */
static NPY_INLINE int
_PyArrayNeighborhoodIter_IncrCoord(PyArrayNeighborhoodIterObject* iter);

/*
 * Update to next item of the iterator
 *
 * Note: this simply increment the coordinates vector, last dimension
 * incremented first , i.e, for dimension 3
 * ...
 * -1, -1, -1
 * -1, -1,  0
 * -1, -1,  1
 *  ....
 * -1,  0, -1
 * -1,  0,  0
 *  ....
 * 0,  -1, -1
 * 0,  -1,  0
 *  ....
 */
#define _UPDATE_COORD_ITER(c) \
    wb = iter->coordinates[c] < iter->bounds[c][1]; \
    if (wb) { \
        iter->coordinates[c] += 1; \
        return 0; \
    } \
    else { \
        iter->coordinates[c] = iter->bounds[c][0]; \
    }

static NPY_INLINE int
_PyArrayNeighborhoodIter_IncrCoord(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp i, wb;

    for (i = iter->nd - 1; i >= 0; --i) {
        _UPDATE_COORD_ITER(i)
    }

    return 0;
}

/*
 * Version optimized for 2d arrays, manual loop unrolling
 */
static NPY_INLINE int
_PyArrayNeighborhoodIter_IncrCoord2D(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp wb;

    _UPDATE_COORD_ITER(1)
    _UPDATE_COORD_ITER(0)

    return 0;
}
#undef _UPDATE_COORD_ITER

/*
 * Advance to the next neighbour
 */
static NPY_INLINE int
PyArrayNeighborhoodIter_Next(PyArrayNeighborhoodIterObject* iter)
{
    _PyArrayNeighborhoodIter_IncrCoord (iter);
    iter->dataptr = iter->translate((PyArrayIterObject*)iter, iter->coordinates);

    return 0;
}

/*
 * Reset functions
 */
static NPY_INLINE int
PyArrayNeighborhoodIter_Reset(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp i;

    for (i = 0; i < iter->nd; ++i) {
        iter->coordinates[i] = iter->bounds[i][0];
    }
    iter->dataptr = iter->translate((PyArrayIterObject*)iter, iter->coordinates);

    return 0;
}
