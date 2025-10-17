/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

#ifndef FIDDLE_CONVERSIONS_H
#define FIDDLE_CONVERSIONS_H

#include <fiddle.h>

typedef union
{
    ffi_arg  fffi_arg;     /* rvalue smaller than unsigned long */
    ffi_sarg fffi_sarg;    /* rvalue smaller than signed long */
    unsigned char uchar;   /* ffi_type_uchar */
    signed char   schar;   /* ffi_type_schar */
    unsigned short ushort; /* ffi_type_sshort */
    signed short sshort;   /* ffi_type_ushort */
    unsigned int uint;     /* ffi_type_uint */
    signed int sint;       /* ffi_type_sint */
    unsigned long ulong;   /* ffi_type_ulong */
    signed long slong;     /* ffi_type_slong */
    float ffloat;          /* ffi_type_float */
    double ddouble;        /* ffi_type_double */
#if HAVE_LONG_LONG
    unsigned LONG_LONG ulong_long; /* ffi_type_ulong_long */
    signed LONG_LONG slong_long; /* ffi_type_ulong_long */
#endif
    void * pointer;        /* ffi_type_pointer */
} fiddle_generic;

ffi_type * int_to_ffi_type(int type);
void value_to_generic(int type, VALUE src, fiddle_generic * dst);
VALUE generic_to_value(VALUE rettype, fiddle_generic retval);

#define VALUE2GENERIC(_type, _src, _dst) value_to_generic((_type), (_src), (_dst))
#define INT2FFI_TYPE(_type) int_to_ffi_type(_type)
#define GENERIC2VALUE(_type, _retval) generic_to_value((_type), (_retval))

#if SIZEOF_VOIDP == SIZEOF_LONG
# define PTR2NUM(x)   (LONG2NUM((long)(x)))
# define NUM2PTR(x)   ((void*)(NUM2ULONG(x)))
#else
/* # error --->> Ruby/DL2 requires sizeof(void*) == sizeof(long) to be compiled. <<--- */
# define PTR2NUM(x)   (LL2NUM((LONG_LONG)(x)))
# define NUM2PTR(x)   ((void*)(NUM2ULL(x)))
#endif

#endif
