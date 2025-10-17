/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
#ifndef LIBFFI_TARGET_H
#define LIBFFI_TARGET_H

#ifndef LIBFFI_H
#error "Please do not include ffitarget.h directly into your source.  Use ffi.h instead."
#endif

/* ---- System specific configurations ----------------------------------- */

#if defined (POWERPC) && defined (__powerpc64__)	/* linux64 */
#ifndef POWERPC64
#define POWERPC64
#endif
#elif defined (POWERPC_DARWIN) && defined (__ppc64__)	/* Darwin64 */
#ifndef POWERPC64
#define POWERPC64
#endif
#ifndef POWERPC_DARWIN64
#define POWERPC_DARWIN64
#endif
#elif defined (POWERPC_AIX) && defined (__64BIT__)	/* AIX64 */
#ifndef POWERPC64
#define POWERPC64
#endif
#endif

#ifndef LIBFFI_ASM
typedef unsigned long          ffi_arg;
typedef signed long            ffi_sarg;

typedef enum ffi_abi {
  FFI_FIRST_ABI = 0,

#if defined (POWERPC_AIX)
  FFI_AIX,
  FFI_DARWIN,
  FFI_DEFAULT_ABI = FFI_AIX,
  FFI_LAST_ABI

#elif defined (POWERPC_DARWIN)
  FFI_AIX,
  FFI_DARWIN,
  FFI_DEFAULT_ABI = FFI_DARWIN,
  FFI_LAST_ABI

#else
  /* The FFI_COMPAT values are used by old code.  Since libffi may be
     a shared library we have to support old values for backwards
     compatibility.  */
  FFI_COMPAT_SYSV,
  FFI_COMPAT_GCC_SYSV,
  FFI_COMPAT_LINUX64,
  FFI_COMPAT_LINUX,
  FFI_COMPAT_LINUX_SOFT_FLOAT,

# if defined (POWERPC64)
  /* This bit, always set in new code, must not be set in any of the
     old FFI_COMPAT values that might be used for 64-bit linux.  We
     only need worry about FFI_COMPAT_LINUX64, but to be safe avoid
     all old values.  */
  FFI_LINUX = 8,
  /* This and following bits can reuse FFI_COMPAT values.  */
  FFI_LINUX_STRUCT_ALIGN = 1,
  FFI_LINUX_LONG_DOUBLE_128 = 2,
  FFI_DEFAULT_ABI = (FFI_LINUX
#  ifdef __STRUCT_PARM_ALIGN__
		     | FFI_LINUX_STRUCT_ALIGN
#  endif
#  ifdef __LONG_DOUBLE_128__
		     | FFI_LINUX_LONG_DOUBLE_128
#  endif
		     ),
  FFI_LAST_ABI = 12

# else
  /* This bit, always set in new code, must not be set in any of the
     old FFI_COMPAT values that might be used for 32-bit linux/sysv/bsd.  */
  FFI_SYSV = 8,
  /* This and following bits can reuse FFI_COMPAT values.  */
  FFI_SYSV_SOFT_FLOAT = 1,
  FFI_SYSV_STRUCT_RET = 2,
  FFI_SYSV_IBM_LONG_DOUBLE = 4,
  FFI_SYSV_LONG_DOUBLE_128 = 16,

  FFI_DEFAULT_ABI = (FFI_SYSV
#  ifdef __NO_FPRS__
		     | FFI_SYSV_SOFT_FLOAT
#  endif
#  if (defined (__SVR4_STRUCT_RETURN)					\
       || defined (POWERPC_FREEBSD) && !defined (__AIX_STRUCT_RETURN))
		     | FFI_SYSV_STRUCT_RET
#  endif
#  if __LDBL_MANT_DIG__ == 106
		     | FFI_SYSV_IBM_LONG_DOUBLE
#  endif
#  ifdef __LONG_DOUBLE_128__
		     | FFI_SYSV_LONG_DOUBLE_128
#  endif
		     ),
  FFI_LAST_ABI = 32
# endif
#endif

} ffi_abi;
#endif

/* ---- Definitions for closures ----------------------------------------- */

#define FFI_CLOSURES 1
#define FFI_NATIVE_RAW_API 0
#if defined (POWERPC) || defined (POWERPC_FREEBSD)
# define FFI_TARGET_SPECIFIC_VARIADIC 1
# define FFI_EXTRA_CIF_FIELDS unsigned nfixedargs
#endif

/* For additional types like the below, take care about the order in
   ppc_closures.S. They must follow after the FFI_TYPE_LAST.  */

/* Needed for soft-float long-double-128 support.  */
#define FFI_TYPE_UINT128 (FFI_TYPE_LAST + 1)

/* Needed for FFI_SYSV small structure returns.  */
#define FFI_SYSV_TYPE_SMALL_STRUCT (FFI_TYPE_LAST + 2)

/* Used by ELFv2 for homogenous structure returns.  */
#define FFI_V2_TYPE_FLOAT_HOMOG		(FFI_TYPE_LAST + 1)
#define FFI_V2_TYPE_DOUBLE_HOMOG	(FFI_TYPE_LAST + 2)
#define FFI_V2_TYPE_SMALL_STRUCT	(FFI_TYPE_LAST + 3)

#if _CALL_ELF == 2
# define FFI_TRAMPOLINE_SIZE 32
#else
# if defined(POWERPC64) || defined(POWERPC_AIX)
#  if defined(POWERPC_DARWIN64)
#    define FFI_TRAMPOLINE_SIZE 48
#  else
#    define FFI_TRAMPOLINE_SIZE 24
#  endif
# else /* POWERPC || POWERPC_AIX */
#  define FFI_TRAMPOLINE_SIZE 40
# endif
#endif

#ifndef LIBFFI_ASM
#if defined(POWERPC_DARWIN) || defined(POWERPC_AIX)
struct ffi_aix_trampoline_struct {
    void * code_pointer;	/* Pointer to ffi_closure_ASM */
    void * toc;			/* TOC */
    void * static_chain;	/* Pointer to closure */
};
#endif
#endif

#endif
