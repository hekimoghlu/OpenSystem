/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
/* decSingle.c -- decSingle operations module                         */
/* ------------------------------------------------------------------ */
/* Copyright (c) IBM Corporation, 2000, 2008.  All rights reserved.   */
/*                                                                    */
/* This software is made available under the terms of the             */
/* ICU License -- ICU 1.8.1 and later.                                */
/*                                                                    */
/* The description and User's Guide ("The decNumber C Library") for   */
/* this software is included in the package as decNumber.pdf.  This   */
/* document is also available in HTML, together with specifications,  */
/* testcases, and Web links, on the General Decimal Arithmetic page.  */
/*                                                                    */
/* Please send comments, suggestions, and corrections to the author:  */
/*   mfc@uk.ibm.com                                                   */
/*   Mike Cowlishaw, IBM Fellow                                       */
/*   IBM UK, PO Box 31, Birmingham Road, Warwick CV34 5JL, UK         */
/* ------------------------------------------------------------------ */
/* This module comprises decSingle operations (including conversions) */
/* ------------------------------------------------------------------ */

#include "decContext.h"       // public includes
#include "decSingle.h"        // public includes

/* Constant mappings for shared code */
#define DECPMAX     DECSINGLE_Pmax
#define DECEMIN     DECSINGLE_Emin
#define DECEMAX     DECSINGLE_Emax
#define DECEMAXD    DECSINGLE_EmaxD
#define DECBYTES    DECSINGLE_Bytes
#define DECSTRING   DECSINGLE_String
#define DECECONL    DECSINGLE_EconL
#define DECBIAS     DECSINGLE_Bias
#define DECLETS     DECSINGLE_Declets
#define DECQTINY    (-DECSINGLE_Bias)
// parameters of next-wider format
#define DECWBYTES   DECDOUBLE_Bytes
#define DECWPMAX    DECDOUBLE_Pmax
#define DECWECONL   DECDOUBLE_EconL
#define DECWBIAS    DECDOUBLE_Bias

/* Type and function mappings for shared code */
#define decFloat                   decSingle      // Type name
#define decFloatWider              decDouble      // Type name

// Utility (binary results, extractors, etc.)
#define decFloatFromBCD            decSingleFromBCD
#define decFloatFromPacked         decSingleFromPacked
#define decFloatFromPackedChecked  decSingleFromPackedChecked
#define decFloatFromString         decSingleFromString
#define decFloatFromWider          decSingleFromWider
#define decFloatGetCoefficient     decSingleGetCoefficient
#define decFloatGetExponent        decSingleGetExponent
#define decFloatSetCoefficient     decSingleSetCoefficient
#define decFloatSetExponent        decSingleSetExponent
#define decFloatShow               decSingleShow
#define decFloatToBCD              decSingleToBCD
#define decFloatToEngString        decSingleToEngString
#define decFloatToPacked           decSingleToPacked
#define decFloatToString           decSingleToString
#define decFloatToWider            decSingleToWider
#define decFloatZero               decSingleZero

// Non-computational
#define decFloatRadix              decSingleRadix
#define decFloatVersion            decSingleVersion

#include "decNumberLocal.h"   // local includes (need DECPMAX)
#include "decCommon.c"        // non-basic decFloat routines
// [Do not include decBasic.c for decimal32]

