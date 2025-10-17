/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
 * Natural alignment of shorts and longs (for i386)
 *
 * HISTORY
 *
 * 2 Sept 1992 Brian Raymor at NeXT
 *      Moved over to architecture.
 * 18 August 1992 Jack Greenfield at NeXT
 *	Created.
 */

#ifndef _ARCH_I386_ALIGNMENT_H_
#define _ARCH_I386_ALIGNMENT_H_

/*
 * NOP
 */
__inline__ static unsigned short
get_align_short(void *ivalue)
{
    return *((unsigned short *) ivalue);
}

__inline__ static unsigned short
put_align_short(unsigned short ivalue, void *ovalue)
{
    return *((unsigned short *) ovalue) = ivalue;
}

/*
 * NOP
 */
__inline__ static unsigned long
get_align_long(void *ivalue)
{
    return *((unsigned long *) ivalue);
}

__inline__ static unsigned long
put_align_long(unsigned long ivalue, void *ovalue)
{
    return *((unsigned long *) ovalue) = ivalue;
}

#endif	/* _ARCH_I386_ALIGNMENT_H_ */
