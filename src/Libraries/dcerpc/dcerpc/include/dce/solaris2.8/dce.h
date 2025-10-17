/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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
#if !defined(_DCE_H)
#define _DCE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Common definitions for DCE
 * This is a machine specific file that must be ported to each platform.
 */

#define DCE_VERSION "1.1"
#define DCE_MAJOR_VERSION 1
#define DCE_MINOR_VERSION 1

/*
 * Define the endianess of the platform. Pulled in from machine/endian.h.
 */

#include <sys/types.h>
#include <netinet/in.h>
#include <inttypes.h>
#include <sys/byteorder.h>

/* Only one place needed for DCE to define these */
#define FALSE 0
#define TRUE 1

#if !defined(MIN)
#  define MIN(x, y)         ((x) < (y) ? (x) : (y))
#endif

#if !defined(MAX)
#  define MAX(x, y)         ((x) > (y) ? (x) : (y))
#endif

/*
 * The following allows for the support of both old and new style
 * function definitions and prototypes.  All DCE code is required to
 * be ANSI C compliant and to use prototypes.  For those components
 * that wish to support old-style definitions, the following macros
 * must be used.
 *
 *  Define a function like this:
 *      int foo
 *              (
 *              int a,
 *              void *b,
 *              struct bar *c
 *              )
 */

/*
 * For those components wishing to support platforms where void
 * pointers are not available, they can use the following typedef for
 * a generic pointer type.  If they are supporting such platforms they
 * must use this.
 */
#if defined(__STDC__)
#  define _DCE_VOID_
#endif                                  /* defined(__STDC__) */

#if defined(_DCE_VOID_)
  typedef void * dce_pointer_t;
#else                                   /* defined(_DCE_VOID_) */
  typedef char * dce_pointer_t;
#endif                                  /* defined(_DCE_VOID_) */

/*
 * Here is a macro that can be used to support token concatenation in
 * an ANSI and non-ANSI environment.  Support of non-ANSI environments
 * is not required, but where done, this macro must be used.
 */
#if defined(__STDC__)
#  define _DCE_TOKENCONCAT_
#endif

#if defined(_DCE_TOKENCONCAT_)
#  define DCE_CONCAT(a, b)      a ## b
#else                                   /* defined(_DCE_TOKENCONCAT_) */
#  define DCE_CONCAT(a, b)      a/**/b
#endif                                  /* defined(_DCE_TOKENCONCAT_) */

/*
 * Define the dcelocal and dceshared directories
 */
extern const char *dcelocal_path;
extern const char *dceshared_path;

/* If DCE_DEBUG is defined then debugging code is activated. */
/* #define DCE_DEBUG */

/*
 * Machine dependent typedefs for boolean, byte, and (un)signed integers.
 * All DCE code should be using these typedefs where applicable.
 * The following are defined in nbase.h:
 *     unsigned8       unsigned  8 bit integer
 *     unsigned16      unsigned 16 bit integer
 *     unsigned32      unsigned 32 bit integer
 *     signed8           signed  8 bit integer
 *     signed16          signed 16 bit integer
 *     signed32          signed 32 bit integer
 * Define the following from idl types in idlbase.h (which is included
 * by nbase.h:
 *     byte            unsigned  8 bits
 *     boolean         unsigned  8 bits
 * Define (un)signed64 to be used with the U64* macros
 */
#include <dce/nbase.h>
typedef idl_byte        byte;
typedef idl_boolean     boolean;
typedef struct unsigned64_s_t {
    unsigned long hi;
    unsigned long lo;
} unsigned64;

typedef struct signed64_s_t {
    unsigned long hi;
    unsigned long lo;
} signed64;

typedef struct unsigned48_s_t {
    unsigned long  int  lo;             /* least significant 32 bits */
	unsigned short int  hi;             /* most significant 16 bits */
} unsigned48;

typedef struct unsigned128_s_t {
    unsigned long lolo;
    unsigned long lohi;
    unsigned long hilo;
    unsigned long hihi;
} unsigned128;

/*
 * Serviceability and perhaps other DCE-wide include files
 * will be included here.  This is a sample only.
 */
#if 0
#include <dce/dce_svc.h>
#endif

#ifdef __cplusplus
}
#endif

#endif                                  /* _DCE_H */
