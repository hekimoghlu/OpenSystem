/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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
 * Intel386 Family:	Definition of eflags register.
 *
 */

#if     KERNEL_PRIVATE

#ifndef _BSD_I386_PSL_H_
#define _BSD_I386_PSL_H_

#define EFL_ALLCC       (               \
	                    EFL_CF |    \
	                    EFL_PF |    \
	                    EFL_AF |    \
	                    EFL_ZF |    \
	                    EFL_SF |    \
	                    EFL_OF      \
	                )
#define EFL_USERSET     ( EFL_IF | EFL_SET )
#define EFL_USERCLR     ( EFL_VM | EFL_NT | EFL_IOPL | EFL_CLR )

#define PSL_ALLCC       EFL_ALLCC
#define PSL_T           EFL_TF

#endif  /* _BSD_I386_PSL_H_ */

#endif  /* KERNEL_PRIVATE */
