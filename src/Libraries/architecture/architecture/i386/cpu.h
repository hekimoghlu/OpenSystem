/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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
 * Copyright (c) 1992 NeXT Computer, Inc.
 *
 * Intel386 Family:	Special processor registers.
 *
 * HISTORY
 *
 * 5 April 1992 ? at NeXT
 *	Created.
 */

#ifndef _ARCH_I386_CPU_H_
#define _ARCH_I386_CPU_H_

/*
 * Control register 0
 */

typedef struct _cr0 {
    unsigned int	pe	:1,
    			mp	:1,
			em	:1,
			ts	:1,
				:1,
			ne	:1,
				:10,
			wp	:1,
				:1,
			am	:1,
				:10,
			nw	:1,
			cd	:1,
			pg	:1;
} cr0_t;

/*
 * Debugging register 6
 */

typedef struct _dr6 {
    unsigned int	b0	:1,
    			b1	:1,
			b2	:1,
			b3	:1,
				:9,
			bd	:1,
			bs	:1,
			bt	:1,
				:16;
} dr6_t;

#endif	/* _ARCH_I386_CPU_H_ */
