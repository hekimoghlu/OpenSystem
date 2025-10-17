/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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
typedef char               mp_sign;
typedef unsigned short     mp_digit;  /* 2 byte type */
typedef unsigned int       mp_word;   /* 4 byte type */
typedef unsigned int       mp_size;
typedef int                mp_err;

#define MP_DIGIT_BIT       (CHAR_BIT*sizeof(mp_digit))
#define MP_DIGIT_MAX       USHRT_MAX
#define MP_WORD_BIT        (CHAR_BIT*sizeof(mp_word))
#define MP_WORD_MAX        UINT_MAX

#define MP_DIGIT_SIZE      2
#define DIGIT_FMT          "%04X"
#define RADIX              (MP_DIGIT_MAX+1)


/* $Source: /cvs/libtom/libtommath/mtest/mpi-types.h,v $ */
/* $Revision: 1.2 $ */
/* $Date: 2005/05/05 14:38:47 $ */
