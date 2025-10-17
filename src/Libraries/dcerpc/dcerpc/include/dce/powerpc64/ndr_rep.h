/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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
 * powerpc/ndr_rep.h
 *
 * platform dependent (OS + Architecture) file split out from stubbase.h
 * for DCE 1.1 code cleanup.
 *
 * For POWER architecture - 64bit, Big Endian Mode
 *
 * This file contains the architecture specific definitions of the
 * local scalar data representation used
 *
 * This file is always included as part of stubbase.h
 */

#ifndef _NDR_REP_H
#define _NDR_REP_H

#define NDR_LOCAL_INT_REP     ndr_c_int_big_endian
#define NDR_LOCAL_FLOAT_REP   ndr_c_float_ieee
#define NDR_LOCAL_CHAR_REP    ndr_c_char_ascii

#endif /* _NDR_REP_H */
