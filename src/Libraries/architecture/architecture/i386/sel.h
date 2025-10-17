/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
 * Intel386 Family:	Segment selector.
 *
 * HISTORY
 *
 * 29 March 1992 ? at NeXT
 *	Created.
 */

#ifndef	_ARCH_I386_SEL_H_
#define	_ARCH_I386_SEL_H_

/*
 * Segment selector.
 */

typedef struct sel {
    unsigned short	rpl	:2,
#define KERN_PRIV	0
#define USER_PRIV	3
			ti	:1,
#define SEL_GDT		0
#define SEL_LDT		1
			index	:13;
} sel_t;

#define NULL_SEL	((sel_t) { 0, 0, 0 } )

#endif	/* _ARCH_I386_SEL_H_ */
