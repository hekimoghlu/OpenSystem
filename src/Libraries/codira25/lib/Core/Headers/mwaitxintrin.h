/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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
#ifndef __X86INTRIN_H
#error "Never use <mwaitxintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __MWAITXINTRIN_H
#define __MWAITXINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__,  __target__("mwaitx")))

/// Establishes a linear address memory range to be monitored and puts
///    the processor in the monitor event pending state. Data stored in the
///    monitored address range causes the processor to exit the pending state.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c MONITORX instruction.
///
/// \param __p
///    The memory range to be monitored. The size of the range is determined by
///    CPUID function 0000_0005h.
/// \param __extensions
///    Optional extensions for the monitoring state.
/// \param __hints
///    Optional hints for the monitoring state.
static __inline__ void __DEFAULT_FN_ATTRS
_mm_monitorx(void * __p, unsigned __extensions, unsigned __hints)
{
  __builtin_ia32_monitorx(__p, __extensions, __hints);
}

/// Used with the \c MONITORX instruction to wait while the processor is in
///    the monitor event pending state. Data stored in the monitored address
///    range, or an interrupt, causes the processor to exit the pending state.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c MWAITX instruction.
///
/// \param __extensions
///    Optional extensions for the monitoring state, which can vary by
///    processor.
/// \param __hints
///    Optional hints for the monitoring state, which can vary by processor.
static __inline__ void __DEFAULT_FN_ATTRS
_mm_mwaitx(unsigned __extensions, unsigned __hints, unsigned __clock)
{
  __builtin_ia32_mwaitx(__extensions, __hints, __clock);
}

#undef __DEFAULT_FN_ATTRS

#endif /* __MWAITXINTRIN_H */
