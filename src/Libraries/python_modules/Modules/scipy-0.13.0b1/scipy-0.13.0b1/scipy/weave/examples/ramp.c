/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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

#include <time.h>
#include <stdio.h>

void Ramp(double* result, int size, double start, double end)
{
    double step = (end-start)/(size-1);
    double val = start;
    int i;
    for (i = 0; i < size; i++)
    {
        *result++ = val;
        val += step;           
    }    
}

void main()
{
    double array[10000];
    int i;
    clock_t t1, t2;
    float seconds;
    t1 = clock();
    for (i = 0; i < 10000; i++)
        Ramp(array, 10000, 0.0, 1.0);
    t2 = clock();
    seconds = (float)(t2-t1)/CLOCKS_PER_SEC; 
    printf("c version (seconds): %f\n", seconds);
    printf("array[500]: %f\n", array[500]);
}