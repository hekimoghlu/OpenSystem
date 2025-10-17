/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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
 * Copyright (c) 1987 Next, Inc.
 *
 * HISTORY
 * 23-Jan-93  Doug Mitchell at NeXT
 *	Broke out machine-independent portion.
 */

#ifdef  DRIVER_PRIVATE

#ifndef _BUSVAR_
#define _BUSVAR_

/* pseudo device initialization routine support */
struct pseudo_init {
	int     ps_count;
	int     (*ps_func)(int count);
};
extern struct pseudo_init pseudo_inits[];

#endif /* _BUSVAR_ */

#endif  /* DRIVER_PRIVATE */
