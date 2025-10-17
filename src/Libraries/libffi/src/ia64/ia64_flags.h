/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
/* "Type" codes used between assembly and C.  When used as a part of
   a cfi->flags value, the low byte will be these extra type codes,
   and bits 8-31 will be the actual size of the type.  */

/* Small structures containing N words in integer registers.  */
#define FFI_IA64_TYPE_SMALL_STRUCT	(FFI_TYPE_LAST + 1)

/* Homogeneous Floating Point Aggregates (HFAs) which are returned
   in FP registers.  */
#define FFI_IA64_TYPE_HFA_FLOAT		(FFI_TYPE_LAST + 2)
#define FFI_IA64_TYPE_HFA_DOUBLE	(FFI_TYPE_LAST + 3)
#define FFI_IA64_TYPE_HFA_LDOUBLE	(FFI_TYPE_LAST + 4)
