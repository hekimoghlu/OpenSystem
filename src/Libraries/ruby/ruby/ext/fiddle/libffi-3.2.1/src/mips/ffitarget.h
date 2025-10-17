/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 6, 2023.
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

#ifdef linux
# include <asm/sgidefs.h>
#elif defined(__rtems__)
/*
 * Subprogram calling convention - copied from sgidefs.h
 */
#define _MIPS_SIM_ABI32		1
#define _MIPS_SIM_NABI32	2
#define _MIPS_SIM_ABI64		3
#elif !defined(__OpenBSD__)
# include <sgidefs.h>
#endif

#  ifndef _ABIN32
#    define _ABIN32 _MIPS_SIM_NABI32
#  endif
#  ifndef _ABI64
#    define _ABI64 _MIPS_SIM_ABI64
#  endif
#  ifndef _ABIO32
#    define _ABIO32 _MIPS_SIM_ABI32
#  endif

#if !defined(_MIPS_SIM)
# error -- something is very wrong --
#else
#  if (_MIPS_SIM==_ABIN32 && defined(_ABIN32)) || (_MIPS_SIM==_ABI64 && defined(_ABI64))
#    define FFI_MIPS_N32
#  else
#    if (_MIPS_SIM==_ABIO32 && defined(_ABIO32))
#      define FFI_MIPS_O32
#    else
#     error -- this is an unsupported platform --
#    endif
#  endif
#endif

#ifdef FFI_MIPS_O32
/* O32 stack frames have 32bit integer args */
#  define FFI_SIZEOF_ARG    4
#else
/* N32 and N64 frames have 64bit integer args */
#  define FFI_SIZEOF_ARG    8
#  if _MIPS_SIM == _ABIN32
#    define FFI_SIZEOF_JAVA_RAW  4
#  endif
#endif

#define FFI_FLAG_BITS 2

/* SGI's strange assembler requires that we multiply by 4 rather 
   than shift left by FFI_FLAG_BITS */

#define FFI_ARGS_D   FFI_TYPE_DOUBLE
#define FFI_ARGS_F   FFI_TYPE_FLOAT
#define FFI_ARGS_DD  FFI_TYPE_DOUBLE * 4 + FFI_TYPE_DOUBLE
#define FFI_ARGS_FF  FFI_TYPE_FLOAT * 4 +  FFI_TYPE_FLOAT
#define FFI_ARGS_FD  FFI_TYPE_DOUBLE * 4 + FFI_TYPE_FLOAT
#define FFI_ARGS_DF  FFI_TYPE_FLOAT * 4 + FFI_TYPE_DOUBLE

/* Needed for N32 structure returns */
#define FFI_TYPE_SMALLSTRUCT  FFI_TYPE_UINT8
#define FFI_TYPE_SMALLSTRUCT2 FFI_TYPE_SINT8

#if 0
/* The SGI assembler can't handle this.. */
#define FFI_TYPE_STRUCT_DD (( FFI_ARGS_DD ) << 4) + FFI_TYPE_STRUCT
/* (and so on) */
#else
/* ...so we calculate these by hand! */
#define FFI_TYPE_STRUCT_D      61
#define FFI_TYPE_STRUCT_F      45
#define FFI_TYPE_STRUCT_DD     253
#define FFI_TYPE_STRUCT_FF     173
#define FFI_TYPE_STRUCT_FD     237
#define FFI_TYPE_STRUCT_DF     189
#define FFI_TYPE_STRUCT_SMALL  93
#define FFI_TYPE_STRUCT_SMALL2 109

/* and for n32 soft float, add 16 * 2^4 */
#define FFI_TYPE_STRUCT_D_SOFT      317
#define FFI_TYPE_STRUCT_F_SOFT      301
#define FFI_TYPE_STRUCT_DD_SOFT     509
#define FFI_TYPE_STRUCT_FF_SOFT     429
#define FFI_TYPE_STRUCT_FD_SOFT     493
#define FFI_TYPE_STRUCT_DF_SOFT     445
#define FFI_TYPE_STRUCT_SOFT        16
#endif

#ifdef LIBFFI_ASM
#define v0 $2
#define v1 $3
#define a0 $4
#define a1 $5
#define a2 $6
#define a3 $7
#define a4 $8		
#define a5 $9		
#define a6 $10		
#define a7 $11		
#define t0 $8
#define t1 $9
#define t2 $10
#define t3 $11
#define t4 $12		
#define t5 $13
#define t6 $14	
#define t7 $15
#define t8 $24
#define t9 $25
#define ra $31		

#ifdef FFI_MIPS_O32
# define REG_L	lw
# define REG_S	sw
# define SUBU	subu
# define ADDU	addu
# define SRL	srl
# define LI	li
#else /* !FFI_MIPS_O32 */
# define REG_L	ld
# define REG_S	sd
# define SUBU	dsubu
# define ADDU	daddu
# define SRL	dsrl
# define LI 	dli
# if (_MIPS_SIM==_ABI64)
#  define LA dla
#  define EH_FRAME_ALIGN 3
#  define FDE_ADDR_BYTES .8byte
# else
#  define LA la
#  define EH_FRAME_ALIGN 2
#  define FDE_ADDR_BYTES .4byte
# endif /* _MIPS_SIM==_ABI64 */
#endif /* !FFI_MIPS_O32 */
#else /* !LIBFFI_ASM */
# ifdef __GNUC__
#  ifdef FFI_MIPS_O32
/* O32 stack frames have 32bit integer args */
typedef unsigned int     ffi_arg __attribute__((__mode__(__SI__)));
typedef signed   int     ffi_sarg __attribute__((__mode__(__SI__)));
#else
/* N32 and N64 frames have 64bit integer args */
typedef unsigned int     ffi_arg __attribute__((__mode__(__DI__)));
typedef signed   int     ffi_sarg __attribute__((__mode__(__DI__)));
#  endif
# else
#  ifdef FFI_MIPS_O32
/* O32 stack frames have 32bit integer args */
typedef __uint32_t ffi_arg;
typedef __int32_t ffi_sarg;
#  else
/* N32 and N64 frames have 64bit integer args */
typedef __uint64_t ffi_arg;
typedef __int64_t ffi_sarg;
#  endif
# endif /* __GNUC__ */

typedef enum ffi_abi {
  FFI_FIRST_ABI = 0,
  FFI_O32,
  FFI_N32,
  FFI_N64,
  FFI_O32_SOFT_FLOAT,
  FFI_N32_SOFT_FLOAT,
  FFI_N64_SOFT_FLOAT,
  FFI_LAST_ABI,

#ifdef FFI_MIPS_O32
#ifdef __mips_soft_float
  FFI_DEFAULT_ABI = FFI_O32_SOFT_FLOAT
#else
  FFI_DEFAULT_ABI = FFI_O32
#endif
#else
# if _MIPS_SIM==_ABI64
#  ifdef __mips_soft_float
  FFI_DEFAULT_ABI = FFI_N64_SOFT_FLOAT
#  else
  FFI_DEFAULT_ABI = FFI_N64
#  endif
# else
#  ifdef __mips_soft_float
  FFI_DEFAULT_ABI = FFI_N32_SOFT_FLOAT
#  else
  FFI_DEFAULT_ABI = FFI_N32
#  endif
# endif
#endif
} ffi_abi;

#define FFI_EXTRA_CIF_FIELDS unsigned rstruct_flag
#endif /* !LIBFFI_ASM */

/* ---- Definitions for closures ----------------------------------------- */

#if defined(FFI_MIPS_O32)
#define FFI_CLOSURES 1
#define FFI_TRAMPOLINE_SIZE 20
#else
/* N32/N64. */
# define FFI_CLOSURES 1
#if _MIPS_SIM==_ABI64
#define FFI_TRAMPOLINE_SIZE 52
#else
#define FFI_TRAMPOLINE_SIZE 20
#endif
#endif /* FFI_MIPS_O32 */
#define FFI_NATIVE_RAW_API 0

#endif

