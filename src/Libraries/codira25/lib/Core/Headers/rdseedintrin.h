/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#ifndef __IMMINTRIN_H
#error "Never use <rdseedintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __RDSEEDINTRIN_H
#define __RDSEEDINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__, __target__("rdseed")))

/// Stores a hardware-generated 16-bit random value in the memory at \a __p.
///
///    The random number generator complies with NIST SP800-90B and SP800-90C.
///
/// \code{.operation}
/// IF HW_NRND_GEN.ready == 1
///   Store16(__p, HW_NRND_GEN.data)
///   result := 1
/// ELSE
///   Store16(__p, 0)
///   result := 0
/// END
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c RDSEED instruction.
///
/// \param __p
///    Pointer to memory for storing the 16-bit random number.
/// \returns 1 if a random number was generated, 0 if not.
static __inline__ int __DEFAULT_FN_ATTRS
_rdseed16_step(unsigned short *__p)
{
  return (int) __builtin_ia32_rdseed16_step(__p);
}

/// Stores a hardware-generated 32-bit random value in the memory at \a __p.
///
///    The random number generator complies with NIST SP800-90B and SP800-90C.
///
/// \code{.operation}
/// IF HW_NRND_GEN.ready == 1
///   Store32(__p, HW_NRND_GEN.data)
///   result := 1
/// ELSE
///   Store32(__p, 0)
///   result := 0
/// END
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c RDSEED instruction.
///
/// \param __p
///    Pointer to memory for storing the 32-bit random number.
/// \returns 1 if a random number was generated, 0 if not.
static __inline__ int __DEFAULT_FN_ATTRS
_rdseed32_step(unsigned int *__p)
{
  return (int) __builtin_ia32_rdseed32_step(__p);
}

#ifdef __x86_64__
/// Stores a hardware-generated 64-bit random value in the memory at \a __p.
///
///    The random number generator complies with NIST SP800-90B and SP800-90C.
///
/// \code{.operation}
/// IF HW_NRND_GEN.ready == 1
///   Store64(__p, HW_NRND_GEN.data)
///   result := 1
/// ELSE
///   Store64(__p, 0)
///   result := 0
/// END
/// \endcode
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the \c RDSEED instruction.
///
/// \param __p
///    Pointer to memory for storing the 64-bit random number.
/// \returns 1 if a random number was generated, 0 if not.
static __inline__ int __DEFAULT_FN_ATTRS
_rdseed64_step(unsigned long long *__p)
{
  return (int) __builtin_ia32_rdseed64_step(__p);
}
#endif

#undef __DEFAULT_FN_ATTRS

#endif /* __RDSEEDINTRIN_H */
