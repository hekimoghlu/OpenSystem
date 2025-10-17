/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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

#define ALPHA_ST_VOID	0
#define ALPHA_ST_INT	1
#define ALPHA_ST_FLOAT	2
#define ALPHA_ST_DOUBLE	3
#define ALPHA_ST_CPLXF	4
#define ALPHA_ST_CPLXD	5

#define ALPHA_LD_VOID	0
#define ALPHA_LD_INT64	1
#define ALPHA_LD_INT32	2
#define ALPHA_LD_UINT16	3
#define ALPHA_LD_SINT16	4
#define ALPHA_LD_UINT8	5
#define ALPHA_LD_SINT8	6
#define ALPHA_LD_FLOAT	7
#define ALPHA_LD_DOUBLE	8
#define ALPHA_LD_CPLXF	9
#define ALPHA_LD_CPLXD	10

#define ALPHA_ST_SHIFT		0
#define ALPHA_LD_SHIFT		8
#define ALPHA_RET_IN_MEM	0x10000
#define ALPHA_FLAGS(S, L)	(((L) << ALPHA_LD_SHIFT) | (S))
