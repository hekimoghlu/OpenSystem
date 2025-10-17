/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
#ifndef _SYS_PROC_REQUIRE_H_
#define _SYS_PROC_REQUIRE_H_

typedef struct proc *   proc_t;

/* Used by proc_require for validation of proc zone */
__options_closed_decl(proc_require_flags_t, unsigned int, {
	PROC_REQUIRE_ALLOW_ALL = 0x0, //always on, allow non null proc
	PROC_REQUIRE_ALLOW_NULL = 0x1,
});

/* validates that 'proc' comes from within the proc zone */
void proc_require(proc_t proc, proc_require_flags_t flags);

#endif // _SYS_PROC_REQUIRE_H_
