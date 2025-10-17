/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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
**
**  NAME
**
**      NIDL.H
**
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**      Mandatory header file containing all system dependent
**      includes and common macros used by the IDL compiler.
**
**  VERSION: DCE 1.0
*/

#ifndef NIDLH_INCL
#define NIDLH_INCL

#define NIDLBASE_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/* Base include files needed by all IDL compiler modules */

#include <stdio.h>
#include <string.h>
#ifdef HAVE_STDBOOL_H
#include <stdbool.h>
#else
typedef enum { false = 0, true = 1 } bool;
#ifndef true
#define true true
#endif
#ifndef false
#define false false
#endif
#endif

#ifdef DUMPERS
# define DEBUG_VERBOSE 1
#endif

#ifdef __STDC__
#   include <stdlib.h>
#   ifndef CHAR_BIT
#       include <limits.h>  /* Bring in limits.h if not cacaded in yet */
#   endif
#else /* prototypes that normally come from stdlib.h */
    extern void *malloc();
    extern void free();
    extern char *getenv();
    extern int atoi();
    extern double atof();
    extern long atol();
#endif
#ifdef __STDC__
#  include <assert.h>
#else
#  define assert(ex) if (ex) ;
#endif
#include <sysdep.h>

/*
 * some generally useful types and macros
 */

typedef unsigned char       unsigned8;
typedef unsigned short int  unsigned16;
typedef unsigned long int   unsigned32;

typedef unsigned8 boolean;
#ifndef TRUE
#define TRUE true
#define FALSE false
#endif

/*
 * IDL's model of the info in a UUID (see idl_uuid_t in nbase.idl)
 */

typedef struct
{
    unsigned32      time_low;
    unsigned16      time_mid;
    unsigned16      time_hi_and_version;
    unsigned8       clock_seq_hi_and_reserved;
    unsigned8       clock_seq_low;
    unsigned8       node[6];
} nidl_uuid_t;

/*
 * Include files needed by the remaining supplied definitions in this file.
 * These need to be here, since they depend on the above definitions.
 */

#include <errors.h>
#include <nidlmsg.h>

/* Language enum.  Here for lack of any place else. */
typedef enum {
    lang_ada_k,
    lang_basic_k,
    lang_c_k,
    lang_cobol_k,
    lang_fortran_k,
    lang_pascal_k
} language_k_t;

/*
 * Macro jackets for each of the C memory management routines.
 * The macros guarantee that control will not return to the caller without
 * memory; therefore the call site doesn't have to test.
 */

/**
 * Returns pointer to a new allocated object of the specified type.
 * It behaves like C++ new. The returned pointer is already correctly typed to
 * type *. So you should not cast it. Let the compiler detect any errors
 * instead of casting.
 *
 * The the returned memory is cleared.
 *
 * @param type of the object that should be allocated
 * @return a valid pointer correctly typed
 */
#define NEW(type)							\
( __extension__								\
	({								\
		type * __local_pointer = calloc(1, sizeof(type));	\
		if (NULL == __local_pointer)				\
			error (NIDL_OUTOFMEM);				\
		__local_pointer;					\
	}))

/**
 * Allocates and returns pointer to a vector of objects.
 * It behaves like C++ new. The returned pointer is already correctly typed to
 * type *. So you should not cast it. Let the compiler detect any errors
 * instead of casting.
 *
 * The the returned memory is cleared.
 *
 * @notice size is the _number_ of objects to be allocated
 *
 * @param type of the object that should be allocated
 * @param size number of objects to be allocated
 * @return a valid pointer correctly typed
 */
#define NEW_VEC(type, size)						\
( __extension__								\
	({								\
		type * __local_pointer = calloc((size), sizeof(type));	\
		if (NULL == __local_pointer)				\
			error (NIDL_OUTOFMEM);				\
		__local_pointer;					\
	}))

/**
 * Reallocates prevoiusly allocated memory area and returns the pointer to it.
 * It behaves like C++ new and C realloc. The returned pointer is already
 * correctly typed to typeof(pointer). So you should not cast it. Let the compiler
 * detect any errors instead of casting.
 *
 * The the returned memory is _not_ cleared.
 *
 * @notice size is the _number_ of objects to be allocated
 *
 * @param pointer points to previously allocated vector
 * @param size number of objects to be allocated
 * @return a valid pointer correctly typed
 */
#define RENEW(pointer, size)							\
( __extension__									\
	({									\
		__typeof__ (pointer) __local_pointer;				\
		__local_pointer =						\
			realloc((pointer),					\
				size * sizeof(__typeof__ (* (pointer))));	\
		if (NULL == __local_pointer)					\
			error (NIDL_OUTOFMEM);					\
		__local_pointer;						\
	}))

/**
 * Allocates some memory area.
 * The returned pointer is always valid. Do not use this function. The better
 * sollution is to use one of the above *NEW* function which return already
 * typed pointers.
 *
 * @param size of the area to be allocated
 * @return a valid pointer to the allocated memory
 */
#define MALLOC(size)						\
( __extension__							\
	({							\
		void * __local_pointer = calloc(1, (size));	\
		if (NULL == __local_pointer)			\
			error (NIDL_OUTOFMEM);			\
		__local_pointer;				\
	}))

/**
 * Frees memory allocated with one of the above functions
 *
 * @param pointer to the memory to be freed
 */
#define FREE(pointer) free (pointer);

/*
 * Enable YYDEBUG, and ASSERTION checking, if DUMPERS is defined
 */
#ifdef DUMPERS
#  define YYDEBUG 1
   /* If ASSERTION expression is FALSE, then issue warning */
#  define ASSERTION(x) do { \
    if (!(x)) { warning(NIDL_INTERNAL_ERROR, __FILE__, __LINE__); } \
} while (0)
#else
#  define ASSERTION(x) do {;} while (0);
#endif

#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
} YYLTYPE;
# define yyltype YYLTYPE /* obsolescent; will be withdrawn */
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif

typedef struct parser_location_t
{
    unsigned	    lineno;
    STRTAB_str_t    fileid;
    YYLTYPE	    location;
    const char *    text;
} parser_location_t;

extern const parser_location_t empty_parser_location;
#define null_parser_location &empty_parser_location

typedef const parser_location_t * parser_location_p;

/* Public NIDL parser API ... */
struct nidl_parser_state_t;
typedef struct nidl_parser_state_t * nidl_parser_p;

nidl_parser_p nidl_parser_alloc (boolean *,void **,char *);
void nidl_parser_destroy (nidl_parser_p);
void nidl_parser_input (nidl_parser_p, FILE *);
unsigned nidl_yylineno (nidl_parser_p);
parser_location_p nidl_location (nidl_parser_p);
unsigned nidl_errcount (nidl_parser_p);
int nidl_yyparse(nidl_parser_p);

/* Public ACF parser API ... */
struct acf_parser_state_t;
typedef struct acf_parser_state_t * acf_parser_p;

acf_parser_p acf_parser_alloc (boolean *, void **, char *);
void acf_parser_input (acf_parser_p, FILE *);
void acf_parser_destroy (acf_parser_p);
unsigned acf_yylineno (acf_parser_p);
parser_location_p acf_location (acf_parser_p);
unsigned acf_errcount (acf_parser_p);
int acf_yyparse (acf_parser_p);

#endif
