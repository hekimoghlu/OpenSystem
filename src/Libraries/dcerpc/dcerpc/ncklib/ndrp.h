/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
**  NAME:
**
**      ndrp.h
**
**  FACILITY:
**
**      Network Data Representation (NDR)
**
**  ABSTRACT:
**
**  System (machine-architecture) -dependent definitions.
**
**
*/

#ifndef _NDRP_H
#define _NDRP_H	1

/*
 * Data representation descriptor (drep)
 *
 * Note that the form of a drep "on the wire" is not captured by the
 * the "ndr_format_t" data type.  The actual structure -- a "packed drep"
 * -- is a vector of four bytes:
 *
 *      | MSB           LSB |
 *      |<---- 8 bits ----->|
 *      |<-- 4 -->|<-- 4 -->|
 *
 *      +---------+---------+
 *      | int rep | chr rep |
 *      +---------+---------+
 *      |     float rep     |
 *      +-------------------+
 *      |     reserved      |
 *      +-------------------+
 *      |     reserved      |
 *      +-------------------+
 *
 * The following macros manipulate data representation descriptors.
 * "NDR_COPY_DREP" copies one packed drep into another.  "NDR_UNPACK_DREP"
 * copies from a packed drep into a variable of the type "ndr_format_t".
 *
 */

#ifdef CONVENTIONAL_ALIGNMENT
#  define NDR_COPY_DREP(dst, src) \
    (*((signed32 *) (dst)) = *((signed32 *) (src)))
#else
#  define NDR_COPY_DREP(dst, src) { \
    (dst)[0] = (src)[0]; \
    (dst)[1] = (src)[1]; \
    (dst)[2] = (src)[2]; \
    (dst)[3] = (src)[3]; \
  }
#endif

#define NDR_DREP_INT_REP(drep)   ((drep)[0] >> 4)
#define NDR_DREP_CHAR_REP(drep)  ((drep)[0] & 0xf)
#define NDR_DREP_FLOAT_REP(drep) ((drep)[1])

#define NDR_UNPACK_DREP(dst, src) {             \
    (dst)->int_rep   = NDR_DREP_INT_REP(src);   \
    (dst)->char_rep  = NDR_DREP_CHAR_REP(src);  \
    (dst)->float_rep = NDR_DREP_FLOAT_REP(src); \
    (dst)->reserved  = 0;                   \
}

#endif /* _NDRP_H */
