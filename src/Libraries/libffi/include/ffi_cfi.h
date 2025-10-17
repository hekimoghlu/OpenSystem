/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
#ifndef FFI_CFI_H
#define FFI_CFI_H

#ifdef HAVE_AS_CFI_PSEUDO_OP

# define cfi_startproc			.cfi_startproc
# define cfi_endproc			.cfi_endproc
# define cfi_def_cfa(reg, off)		.cfi_def_cfa reg, off
# define cfi_def_cfa_register(reg)	.cfi_def_cfa_register reg
# define cfi_def_cfa_offset(off)	.cfi_def_cfa_offset off
# define cfi_adjust_cfa_offset(off)	.cfi_adjust_cfa_offset off
# define cfi_offset(reg, off)		.cfi_offset reg, off
# define cfi_rel_offset(reg, off)	.cfi_rel_offset reg, off
# define cfi_register(r1, r2)		.cfi_register r1, r2
# define cfi_return_column(reg)		.cfi_return_column reg
# define cfi_restore(reg)		.cfi_restore reg
# define cfi_same_value(reg)		.cfi_same_value reg
# define cfi_undefined(reg)		.cfi_undefined reg
# define cfi_remember_state		.cfi_remember_state
# define cfi_restore_state		.cfi_restore_state
# define cfi_window_save		.cfi_window_save
# define cfi_personality(enc, exp)	.cfi_personality enc, exp
# define cfi_lsda(enc, exp)		.cfi_lsda enc, exp
# define cfi_escape(...)		.cfi_escape __VA_ARGS__

#else

# define cfi_startproc
# define cfi_endproc
# define cfi_def_cfa(reg, off)
# define cfi_def_cfa_register(reg)
# define cfi_def_cfa_offset(off)
# define cfi_adjust_cfa_offset(off)
# define cfi_offset(reg, off)
# define cfi_rel_offset(reg, off)
# define cfi_register(r1, r2)
# define cfi_return_column(reg)
# define cfi_restore(reg)
# define cfi_same_value(reg)
# define cfi_undefined(reg)
# define cfi_remember_state
# define cfi_restore_state
# define cfi_window_save
# define cfi_personality(enc, exp)
# define cfi_lsda(enc, exp)
# define cfi_escape(...)

#endif /* HAVE_AS_CFI_PSEUDO_OP */
#endif /* FFI_CFI_H */
