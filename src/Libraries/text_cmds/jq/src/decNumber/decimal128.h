/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
/* Decimal 128-bit format module header                               */
/* ------------------------------------------------------------------ */
/* Copyright (c) IBM Corporation, 2000, 2005.  All rights reserved.   */
/*                                                                    */
/* This software is made available under the terms of the             */
/* ICU License -- ICU 1.8.1 and later.                                */
/*                                                                    */
/* The description and User's Guide ("The decNumber C Library") for   */
/* this software is called decNumber.pdf.  This document is           */
/* available, together with arithmetic and format specifications,     */
/* testcases, and Web links, on the General Decimal Arithmetic page.  */
/*                                                                    */
/* Please send comments, suggestions, and corrections to the author:  */
/*   mfc@uk.ibm.com                                                   */
/*   Mike Cowlishaw, IBM Fellow                                       */
/*   IBM UK, PO Box 31, Birmingham Road, Warwick CV34 5JL, UK         */
/* ------------------------------------------------------------------ */

#if !defined(DECIMAL128)
  #define DECIMAL128
  #define DEC128NAME     "decimal128"                 /* Short name   */
  #define DEC128FULLNAME "Decimal 128-bit Number"     /* Verbose name */
  #define DEC128AUTHOR   "Mike Cowlishaw"             /* Who to blame */

  /* parameters for decimal128s */
  #define DECIMAL128_Bytes  16          /* length                     */
  #define DECIMAL128_Pmax   34          /* maximum precision (digits) */
  #define DECIMAL128_Emax   6144        /* maximum adjusted exponent  */
  #define DECIMAL128_Emin  -6143        /* minimum adjusted exponent  */
  #define DECIMAL128_Bias   6176        /* bias for the exponent      */
  #define DECIMAL128_String 43          /* maximum string length, +1  */
  #define DECIMAL128_EconL  12          /* exp. continuation length   */
  /* highest biased exponent (Elimit-1)                               */
  #define DECIMAL128_Ehigh  (DECIMAL128_Emax+DECIMAL128_Bias-DECIMAL128_Pmax+1)

  /* check enough digits, if pre-defined                              */
  #if defined(DECNUMDIGITS)
    #if (DECNUMDIGITS<DECIMAL128_Pmax)
      #error decimal128.h needs pre-defined DECNUMDIGITS>=34 for safe use
    #endif
  #endif

  #ifndef DECNUMDIGITS
    #define DECNUMDIGITS DECIMAL128_Pmax /* size if not already defined*/
  #endif
  #ifndef DECNUMBER
    #include "decNumber.h"              /* context and number library */
  #endif

  /* Decimal 128-bit type, accessible by bytes                        */
  typedef struct {
    uint8_t bytes[DECIMAL128_Bytes]; /* decimal128: 1, 5, 12, 110 bits*/
    } decimal128;

  /* special values [top byte excluding sign bit; last two bits are   */
  /* don't-care for Infinity on input, last bit don't-care for NaN]   */
  #if !defined(DECIMAL_NaN)
    #define DECIMAL_NaN     0x7c        /* 0 11111 00 NaN             */
    #define DECIMAL_sNaN    0x7e        /* 0 11111 10 sNaN            */
    #define DECIMAL_Inf     0x78        /* 0 11110 00 Infinity        */
  #endif

  /* ---------------------------------------------------------------- */
  /* Routines                                                         */
  /* ---------------------------------------------------------------- */
  /* String conversions                                               */
  decimal128 * decimal128FromString(decimal128 *, const char *, decContext *);
  char * decimal128ToString(const decimal128 *, char *);
  char * decimal128ToEngString(const decimal128 *, char *);

  /* decNumber conversions                                            */
  decimal128 * decimal128FromNumber(decimal128 *, const decNumber *,
                                    decContext *);
  decNumber * decimal128ToNumber(const decimal128 *, decNumber *);

  /* Format-dependent utilities                                       */
  uint32_t    decimal128IsCanonical(const decimal128 *);
  decimal128 * decimal128Canonical(decimal128 *, const decimal128 *);

#endif
