/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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

#ifndef SUPERTENSOR_H
#define SUPERTENSOR_H

// The following macro defines the prototypes for a family of
// functions that work with 4D arrays with the forms
//
//     TYPE SNAMENorm(   TYPE supertensor[2][2][2][2]);
//     TYPE SNAMEMax(    TYPE * supertensor, int cubes, int slices, int rows, int cols);
//     TYPE SNAMEMin(    int cubes, int slices, int rows, int cols, TYPE * supertensor);
//     void SNAMEScale(  TYPE array[3][3][3][3]);
//     void SNAMEFloor(  TYPE * array,  int cubes, int slices, int rows, int cols, TYPE floor);
//     void SNAMECeil(   int cubes, int slices, int rows, int cols, TYPE * array,  TYPE ceil );
//     void SNAMELUSplit(TYPE in[3][3][3][3], TYPE lower[3][3][3][3], TYPE upper[3][3][3][3]);
//
// for any specified type TYPE (for example: short, unsigned int, long
// long, etc.) with given short name SNAME (for example: short, uint,
// longLong, etc.).  The macro is then expanded for the given
// TYPE/SNAME pairs.  The resulting functions are for testing numpy
// interfaces, respectively, for:
//
//  * 4D input arrays, hard-coded lengths
//  * 4D input arrays
//  * 4D input arrays, data last
//  * 4D in-place arrays, hard-coded lengths
//  * 4D in-place arrays
//  * 4D in-place arrays, data last
//  * 4D argout arrays, hard-coded length
//
#define TEST_FUNC_PROTOS(TYPE, SNAME) \
\
TYPE SNAME ## Norm(   TYPE supertensor[2][2][2][2]); \
TYPE SNAME ## Max(    TYPE * supertensor, int cubes, int slices, int rows, int cols); \
TYPE SNAME ## Min(    int cubes, int slices, int rows, int cols, TYPE * supertensor); \
void SNAME ## Scale(  TYPE array[3][3][3][3], TYPE val); \
void SNAME ## Floor(  TYPE * array, int cubes, int slices, int rows, int cols, TYPE floor); \
void SNAME ## Ceil(   int cubes, int slices, int rows, int cols, TYPE * array, TYPE ceil ); \
void SNAME ## LUSplit(TYPE supertensor[2][2][2][2], TYPE lower[2][2][2][2], TYPE upper[2][2][2][2]);

TEST_FUNC_PROTOS(signed char       , schar    )
TEST_FUNC_PROTOS(unsigned char     , uchar    )
TEST_FUNC_PROTOS(short             , short    )
TEST_FUNC_PROTOS(unsigned short    , ushort   )
TEST_FUNC_PROTOS(int               , int      )
TEST_FUNC_PROTOS(unsigned int      , uint     )
TEST_FUNC_PROTOS(long              , long     )
TEST_FUNC_PROTOS(unsigned long     , ulong    )
TEST_FUNC_PROTOS(long long         , longLong )
TEST_FUNC_PROTOS(unsigned long long, ulongLong)
TEST_FUNC_PROTOS(float             , float    )
TEST_FUNC_PROTOS(double            , double   )

#endif

