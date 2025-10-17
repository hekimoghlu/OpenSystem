/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
#ifndef __CET_H
#define __CET_H

#ifdef __ASSEMBLER__

#ifndef __CET__
# define _CET_ENDBR
#endif

#ifdef __CET__

# ifdef __LP64__
#  if __CET__ & 0x1
#    define _CET_ENDBR endbr64
#  else
#    define _CET_ENDBR
#  endif
# else
#  if __CET__ & 0x1
#    define _CET_ENDBR endbr32
#  else
#    define _CET_ENDBR
#  endif
# endif


#  ifdef __LP64__
#   define __PROPERTY_ALIGN 3
#  else
#   define __PROPERTY_ALIGN 2
#  endif

	.pushsection ".note.gnu.property", "a"
	.p2align __PROPERTY_ALIGN
	.long 1f - 0f		/* name length.  */
	.long 4f - 1f		/* data length.  */
	/* NT_GNU_PROPERTY_TYPE_0.   */
	.long 5			/* note type.  */
0:
	.asciz "GNU"		/* vendor name.  */
1:
	.p2align __PROPERTY_ALIGN
	/* GNU_PROPERTY_X86_FEATURE_1_AND.  */
	.long 0xc0000002	/* pr_type.  */
	.long 3f - 2f		/* pr_datasz.  */
2:
	/* GNU_PROPERTY_X86_FEATURE_1_XXX.  */
	.long __CET__
3:
	.p2align __PROPERTY_ALIGN
4:
	.popsection
#endif
#endif
#endif
