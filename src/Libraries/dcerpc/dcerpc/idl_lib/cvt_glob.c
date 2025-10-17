/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
/*
**
**
**  NAME:
**
**      cvt_glob.c
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**  Global data used by floating point conversion routines.
**
**  VERSION: DCE 1.0
**
*/
#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <cvt.h>
#include <cvt_pvt.h>

#if (NDR_LOCAL_INT_REP != ndr_c_int_big_endian)

idl_ulong_int vax_c[] = {

        0x00008000, 0x00000000, 0x00000000, 0x00000000,         /* ROPs */
        0x00000000, 0x00000000, 0x00000000, 0x00000000,         /* zeros */
        0xffff7fff, 0xffffffff, 0xffffffff, 0xffffffff,         /* +huge */
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,         /* -huge */

};

idl_ulong_int ieee_s[] = {

        0x7fbfffff,             /* little endian ieee s nan */
        0xffffbf7f,             /* big endian ieee s nan */
        0x00000000,             /* le ieee s +zero */
        0x00000000,             /* be ieee s +zero */
        0x80000000,             /* le ieee s -zero */
        0x00000080,             /* be ieee s -zero */
        0x7f7fffff,             /* le ieee s +huge */
        0xffff7f7f,             /* be ieee s +huge */
        0xff7fffff,             /* le ieee s -huge */
        0xffff7fff,             /* be ieee s -huge */
        0x7f800000,             /* le ieee s +infinity */
        0x0000807f,             /* be ieee s +infinity */
        0xff800000,             /* le ieee s -infinity */
        0x000080ff,             /* be ieee s -infinity */

};

idl_ulong_int ieee_t[] = {

        0xffffffff, 0x7ff7ffff,         /* le ieee t nan */
        0xfffff77f, 0xffffffff,         /* be ieee t nan */
        0x00000000, 0x00000000,         /* le ieee t +zero */
        0x00000000, 0x00000000,         /* be ieee t +zero */
        0x00000000, 0x80000000,         /* le ieee t -zero */
        0x00000080, 0x00000000,         /* be ieee t -zero */
        0xffffffff, 0x7fefffff,         /* le ieee s +huge */
        0xffffef7f, 0xffffffff,         /* be ieee s +huge */
        0xffffffff, 0xffefffff,         /* le ieee s -huge */
        0xffffefff, 0xffffffff,         /* be ieee s -huge */
        0x00000000, 0x7ff00000,         /* le ieee t +infinity */
        0x0000f07f, 0x00000000,         /* be ieee t +infinity */
        0x00000000, 0xfff00000,         /* le ieee t -infinity */
        0x0000f0ff, 0x00000000,         /* be ieee t -infinity */

};

idl_ulong_int ibm_s[] = {

   0x000000ff,          /* ibm s invalid */
   0x00000000,          /* ibm s +zero */
   0x00000080,          /* ibm s -zero */
   0xffffff7f,          /* ibm s +huge */
   0xffffffff,          /* ibm s -huge */
   0xffffff7f,          /* ibm s +infinity */
   0xffffffff,          /* ibm s -infinity */

};

idl_ulong_int ibm_l[] = {

   0x000000ff, 0x00000000,              /* ibm t invalid */
   0x00000000, 0x00000000,              /* ibm t +zero */
   0x00000080, 0x00000000,              /* ibm t -zero */
   0xffffff7f, 0xffffffff,              /* ibm t +huge */
   0xffffffff, 0xffffffff,              /* ibm t -huge */
   0xffffff7f, 0xffffffff,              /* ibm t +infinity */
   0xffffffff, 0xffffffff,              /* ibm t -infinity */

};

idl_ulong_int cray[] = {

        0x00000060, 0x00000000,         /* cray invalid */
        0x00000000, 0x00000000,         /* cray +zero */
        0x00000080, 0x00000000,         /* cray -zero */
        0xffffff5f, 0xffffffff,         /* cray +huge */
        0xffffffdf, 0xffffffff,         /* cray -huge */
        0x00000060, 0x00000000,         /* cray +infinity */
        0x000000e0, 0x00000000,         /* cray -infinity */

};

idl_ulong_int int_c[] = {

        0x00000000,             /* le int nan */
        0x00000000,             /* be int nan */
        0x00000000,             /* le int zero */
        0x00000000,             /* be int zero */
        0x7fffffff,             /* le int +huge */
        0xffffff7f,             /* be int +huge */
        0x80000000,             /* le int -huge */
        0x00000080,             /* be int -huge */
        0x7fffffff,             /* le int +infinity */
        0xffffff7f,             /* be int +infinity */
        0x80000000,             /* le int -infinity */
        0x00000080,             /* be int -infinity */

};

#else /* Big-endian Data */

idl_ulong_int vax_c[] = {

        0x00800000, 0x00000000, 0x00000000, 0x00000000,         /* ROPs */
        0x00000000, 0x00000000, 0x00000000, 0x00000000,         /* zeros */
        0xff7fffff, 0xffffffff, 0xffffffff, 0xffffffff,         /* +huge */
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,         /* -huge */

};

idl_ulong_int ieee_s[] = {

        0xffffbf7f,             /* little endian ieee s nan */
        0x7fbfffff,             /* big endian ieee s nan */
        0x00000000,             /* le ieee s +zero */
        0x00000000,             /* be ieee s +zero */
        0x00000080,             /* le ieee s -zero */
        0x80000000,             /* be ieee s -zero */
        0xffff7f7f,             /* le ieee s +huge */
        0x7f7fffff,             /* be ieee s +huge */
        0xffff7fff,             /* le ieee s -huge */
        0xff7fffff,             /* be ieee s -huge */
        0x0000807f,             /* le ieee s +infinity */
        0x7f800000,             /* be ieee s +infinity */
        0x000080ff,             /* le ieee s -infinity */
        0xff800000,             /* be ieee s -infinity */

};

idl_ulong_int ieee_t[] = {

        0xffffffff, 0xfffff77f,         /* le ieee t nan */
        0x7ff7ffff, 0xffffffff,         /* be ieee t nan */
        0x00000000, 0x00000000,         /* le ieee t +zero */
        0x00000000, 0x00000000,         /* be ieee t +zero */
        0x00000000, 0x00000080,         /* le ieee t -zero */
        0x80000000, 0x00000000,         /* be ieee t -zero */
        0xffffffff, 0xffffef7f,         /* le ieee s +huge */
        0x7fefffff, 0xffffffff,         /* be ieee s +huge */
        0xffffffff, 0xffffefff,         /* le ieee s -huge */
        0xffefffff, 0xffffffff,         /* be ieee s -huge */
        0x00000000, 0x0000f07f,         /* le ieee t +infinity */
        0x7ff00000, 0x00000000,         /* be ieee t +infinity */
        0x00000000, 0x0000f0ff,         /* le ieee t -infinity */
        0xfff00000, 0x00000000,         /* be ieee t -infinity */

};

idl_ulong_int ibm_s[] = {

   0xff000000,          /* ibm s invalid */
   0x00000000,          /* ibm s +zero */
   0x80000000,          /* ibm s -zero */
   0x7fffffff,          /* ibm s +huge */
   0xffffffff,          /* ibm s -huge */
   0x7fffffff,          /* ibm s +infinity */
   0xffffffff,          /* ibm s -infinity */

};

idl_ulong_int ibm_l[] = {

   0xff000000, 0x00000000,              /* ibm t invalid */
   0x00000000, 0x00000000,              /* ibm t +zero */
   0x80000000, 0x00000000,              /* ibm t -zero */
   0x7fffffff, 0xffffffff,              /* ibm t +huge */
   0xffffffff, 0xffffffff,              /* ibm t -huge */
   0x7fffffff, 0xffffffff,              /* ibm t +infinity */
   0xffffffff, 0xffffffff,              /* ibm t -infinity */

};

idl_ulong_int cray[] = {

        0x60000000, 0x00000000,         /* cray invalid */
        0x00000000, 0x00000000,         /* cray +zero */
        0x80000000, 0x00000000,         /* cray -zero */
        0x5fffffff, 0xffffffff,         /* cray +huge */
        0xdfffffff, 0xffffffff,         /* cray -huge */
        0x60000000, 0x00000000,         /* cray +infinity */
        0xe0000000, 0x00000000,         /* cray -infinity */

};

idl_ulong_int int_c[] = {

        0x00000000,             /* le int nan */
        0x00000000,             /* be int nan */
        0x00000000,             /* le int zero */
        0x00000000,             /* be int zero */
        0xffffff7f,             /* le int +huge */
        0x7fffffff,             /* be int +huge */
        0x00000080,             /* le int -huge */
        0x80000000,             /* be int -huge */
        0xffffff7f,             /* le int +infinity */
        0x7fffffff,             /* be int +infinity */
        0x00000080,             /* le int -infinity */
        0x80000000,             /* be int -infinity */

};

#endif /* Big-Endian Data */
