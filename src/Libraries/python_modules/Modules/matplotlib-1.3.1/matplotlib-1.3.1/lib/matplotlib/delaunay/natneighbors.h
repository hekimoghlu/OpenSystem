/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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

#ifndef _NATNEIGHBORS_H
#define _NATNEIGHBORS_H

#include <list>
using namespace std;

class NaturalNeighbors
{
public:
    NaturalNeighbors(int npoints, int ntriangles, double *x, double *y,
        double *centers, int *nodes, int *neighbors);
    ~NaturalNeighbors();

    double interpolate_one(double *z, double targetx, double targety,
        double defvalue, int &start_triangle);

    void interpolate_grid(double *z, 
        double x0, double x1, int xsteps,
        double y0, double y1, int ysteps,
        double *output, double defvalue, int start_triangle);

    void interpolate_unstructured(double *z, int size, 
        double *intx, double *inty, double *output, double defvalue);

private:
    int npoints, ntriangles;
    double *x, *y, *centers, *radii2;
    int *nodes, *neighbors;

    int find_containing_triangle(double targetx, double targety, int start_triangle);
};

#endif // _NATNEIGHBORS_H
