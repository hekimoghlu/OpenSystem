/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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

#define ARM_TYPE_VFP_S	0
#define ARM_TYPE_VFP_D	1
#define ARM_TYPE_VFP_N	2
#define ARM_TYPE_INT64	3
#define ARM_TYPE_INT	4
#define ARM_TYPE_VOID	5
#define ARM_TYPE_STRUCT	6

#if defined(FFI_EXEC_STATIC_TRAMP)
/*
 * For the trampoline table mapping, a mapping size of 4K (base page size)
 * is chosen.
 */
#define ARM_TRAMP_MAP_SHIFT	12
#define ARM_TRAMP_MAP_SIZE	(1 << ARM_TRAMP_MAP_SHIFT)
#define ARM_TRAMP_SIZE		20
#endif
