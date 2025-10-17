/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
 * @OSF_COPYRIGHT@
 */

#ifndef _MACH_NDR_H_
#define _MACH_NDR_H_

#include <stdint.h>
#include <sys/cdefs.h>
#include <libkern/OSByteOrder.h>


typedef struct {
	unsigned char       mig_vers;
	unsigned char       if_vers;
	unsigned char       reserved1;
	unsigned char       mig_encoding;
	unsigned char       int_rep;
	unsigned char       char_rep;
	unsigned char       float_rep;
	unsigned char       reserved2;
} NDR_record_t;

/*
 * MIG supported protocols for Network Data Representation
 */
#define  NDR_PROTOCOL_2_0      0

/*
 * NDR 2.0 format flag type definition and values.
 */
#define  NDR_INT_BIG_ENDIAN    0
#define  NDR_INT_LITTLE_ENDIAN 1
#define  NDR_FLOAT_IEEE        0
#define  NDR_FLOAT_VAX         1
#define  NDR_FLOAT_CRAY        2
#define  NDR_FLOAT_IBM         3
#define  NDR_CHAR_ASCII        0
#define  NDR_CHAR_EBCDIC       1

extern NDR_record_t NDR_record;

/* NDR conversion off by default */

#if !defined(__NDR_convert__)
#define __NDR_convert__ 0
#endif /* !defined(__NDR_convert__) */

#ifndef __NDR_convert__int_rep__
#define __NDR_convert__int_rep__ __NDR_convert__
#endif /* __NDR_convert__int_rep__ */

#ifndef __NDR_convert__char_rep__
#define __NDR_convert__char_rep__ 0
#endif /* __NDR_convert__char_rep__ */

#ifndef __NDR_convert__float_rep__
#define __NDR_convert__float_rep__ 0
#endif /* __NDR_convert__float_rep__ */

#if __NDR_convert__

#define __NDR_convert__NOOP             do ; while (0)
#define __NDR_convert__UNKNOWN(s)       __NDR_convert__NOOP
#define __NDR_convert__SINGLE(a, f, r)  do { r((a), (f)); } while (0)
#define __NDR_convert__ARRAY(a, f, c, r) \
	do { int __i__, __C__ = (c); \
	for (__i__ = 0; __i__ < __C__; __i__++) \
	r(&(a)[__i__], f); } while (0)
#define __NDR_convert__2DARRAY(a, f, s, c, r) \
	do { int __i__, __C__ = (c), __S__ = (s); \
	for (__i__ = 0; __i__ < __C__; __i__++) \
	r(&(a)[__i__ * __S__], f, __S__); } while (0)

#if __NDR_convert__int_rep__

#define __NDR_READSWAP_assign(a, rs)    do { *(a) = rs(a); } while (0)

#define __NDR_READSWAP__uint16_t(a)     OSReadSwapInt16((void *)a, 0)
#define __NDR_READSWAP__int16_t(a)      (int16_t)OSReadSwapInt16((void *)a, 0)
#define __NDR_READSWAP__uint32_t(a)     OSReadSwapInt32((void *)a, 0)
#define __NDR_READSWAP__int32_t(a)      (int32_t)OSReadSwapInt32((void *)a, 0)
#define __NDR_READSWAP__uint64_t(a)     OSReadSwapInt64((void *)a, 0)
#define __NDR_READSWAP__int64_t(a)      (int64_t)OSReadSwapInt64((void *)a, 0)

__BEGIN_DECLS

static __inline__ float
__NDR_READSWAP__float(float *argp)
{
	union {
		float sv;
		uint32_t ull;
	} result;
	result.ull = __NDR_READSWAP__uint32_t((uint32_t *)argp);
	return result.sv;
}

static __inline__ double
__NDR_READSWAP__double(double *argp)
{
	union {
		double sv;
		uint64_t ull;
	} result;
	result.ull = __NDR_READSWAP__uint64_t((uint64_t *)argp);
	return result.sv;
}

__END_DECLS

#define __NDR_convert__int_rep__int16_t__defined
#define __NDR_convert__int_rep__int16_t(v, f)            \
	__NDR_READSWAP_assign(v, __NDR_READSWAP__int16_t)

#define __NDR_convert__int_rep__uint16_t__defined
#define __NDR_convert__int_rep__uint16_t(v, f)           \
	__NDR_READSWAP_assign(v, __NDR_READSWAP__uint16_t)

#define __NDR_convert__int_rep__int32_t__defined
#define __NDR_convert__int_rep__int32_t(v, f)            \
	__NDR_READSWAP_assign(v, __NDR_READSWAP__int32_t)

#define __NDR_convert__int_rep__uint32_t__defined
#define __NDR_convert__int_rep__uint32_t(v, f)           \
	__NDR_READSWAP_assign(v, __NDR_READSWAP__uint32_t)

#define __NDR_convert__int_rep__int64_t__defined
#define __NDR_convert__int_rep__int64_t(v, f)            \
	__NDR_READSWAP_assign(v, __NDR_READSWAP__int64_t)

#define __NDR_convert__int_rep__uint64_t__defined
#define __NDR_convert__int_rep__uint64_t(v, f)           \
	__NDR_READSWAP_assign(v, __NDR_READSWAP__uint64_t)

#define __NDR_convert__int_rep__float__defined
#define __NDR_convert__int_rep__float(v, f)              \
	__NDR_READSWAP_assign(v, __NDR_READSWAP__float)

#define __NDR_convert__int_rep__double__defined
#define __NDR_convert__int_rep__double(v, f)             \
	__NDR_READSWAP_assign(v, __NDR_READSWAP__double)

#define __NDR_convert__int_rep__boolean_t__defined
#define __NDR_convert__int_rep__boolean_t(v, f)         \
	__NDR_convert__int_rep__int32_t(v,f)

#define __NDR_convert__int_rep__kern_return_t__defined
#define __NDR_convert__int_rep__kern_return_t(v, f)      \
	__NDR_convert__int_rep__int32_t(v,f)

#define __NDR_convert__int_rep__mach_port_name_t__defined
#define __NDR_convert__int_rep__mach_port_name_t(v, f)   \
	__NDR_convert__int_rep__uint32_t(v,f)

#define __NDR_convert__int_rep__mach_msg_type_number_t__defined
#define __NDR_convert__int_rep__mach_msg_type_number_t(v, f) \
	__NDR_convert__int_rep__uint32_t(v,f)

#endif /* __NDR_convert__int_rep__ */

#if __NDR_convert__char_rep__

#warning  NDR character representation conversions not implemented yet!
#define __NDR_convert__char_rep__char(v, f)      __NDR_convert__NOOP
#define __NDR_convert__char_rep__string(v, f, l)  __NDR_convert__NOOP

#endif /* __NDR_convert__char_rep__ */

#if __NDR_convert__float_rep__

#warning  NDR floating point representation conversions not implemented yet!
#define __NDR_convert__float_rep__float(v, f)    __NDR_convert__NOOP
#define __NDR_convert__float_rep__double(v, f)   __NDR_convert__NOOP

#endif /* __NDR_convert__float_rep__ */

#endif /* __NDR_convert__ */

#endif /* _MACH_NDR_H_ */
