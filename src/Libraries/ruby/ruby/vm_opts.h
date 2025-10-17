/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
/**********************************************************************

  vm_opts.h - VM optimize option

  $Author: k0kubun $

  Copyright (C) 2004-2007 Koichi Sasada

**********************************************************************/


#ifndef RUBY_VM_OPTS_H
#define RUBY_VM_OPTS_H

/* Compile options.
 * You can change these options at runtime by VM::CompileOption.
 * Following definitions are default values.
 */

#define OPT_TAILCALL_OPTIMIZATION       0
#define OPT_PEEPHOLE_OPTIMIZATION       1
#define OPT_SPECIALISED_INSTRUCTION     1
#define OPT_INLINE_CONST_CACHE          1
#define OPT_FROZEN_STRING_LITERAL       0
#define OPT_DEBUG_FROZEN_STRING_LITERAL 0

/* Build Options.
 * You can't change these options at runtime.
 */

/* C compiler dependent */

/*
 * 0: direct (using labeled goto using GCC special)
 * 1: token (switch/case)
 * 2: call (function call for each insn dispatch)
 */
#ifndef OPT_THREADED_CODE
#define OPT_THREADED_CODE 0
#endif

#define OPT_DIRECT_THREADED_CODE (OPT_THREADED_CODE == 0)
#define OPT_TOKEN_THREADED_CODE  (OPT_THREADED_CODE == 1)
#define OPT_CALL_THREADED_CODE   (OPT_THREADED_CODE == 2)

/* VM running option */
#define OPT_CHECKED_RUN              1
#define OPT_INLINE_METHOD_CACHE      1
#define OPT_GLOBAL_METHOD_CACHE      1
#define OPT_BLOCKINLINING            0

#ifndef OPT_IC_FOR_IVAR
#define OPT_IC_FOR_IVAR 1
#endif

/* architecture independent, affects generated code */
#define OPT_OPERANDS_UNIFICATION     1
#define OPT_INSTRUCTIONS_UNIFICATION 0
#define OPT_UNIFY_ALL_COMBINATION    0
#define OPT_STACK_CACHING            0

/* misc */
#define SUPPORT_JOKE                 0

#ifndef VM_COLLECT_USAGE_DETAILS
#define VM_COLLECT_USAGE_DETAILS     0
#endif

#endif /* RUBY_VM_OPTS_H */
