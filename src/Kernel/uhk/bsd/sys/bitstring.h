/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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
/*-
 * Copyright (c) 1989, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * Paul Vixie.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _SYS_BITSTRING_H_
#define _SYS_BITSTRING_H_

#ifdef XNU_KERNEL_PRIVATE
#include <sys/mcache.h>

typedef uint8_t bitstr_t;

/* internal macros */
/* byte of the bitstring bit is in */
#define _bitstr_byte(bit)                                               \
	((bit) >> 3)

/* mask for the bit within its byte */
#define _bitstr_mask(bit)                                               \
	(1 << ((bit) & 0x7))

/* external macros */
/* bytes in a bitstring of nbits bits */
#define bitstr_size(nbits)                                              \
	(((nbits) + 7) >> 3)

/* allocate a bitstring on the stack */
#define bit_decl(name, nbits)                                           \
	((name)[bitstr_size(nbits)])

/* is bit N of bitstring name set? */
#define bitstr_test(name, bit)                                          \
	((name)[_bitstr_byte(bit)] & _bitstr_mask(bit))

/* set bit N of bitstring name */
#define bitstr_set(name, bit)                                           \
	((name)[_bitstr_byte(bit)] |= _bitstr_mask(bit))

/* set bit N of bitstring name (atomic) */
#define bitstr_set_atomic(name, bit)                                    \
	(void)os_atomic_or(&((name)[_bitstr_byte(bit)]), _bitstr_mask(bit), relaxed)

/* clear bit N of bitstring name */
#define bitstr_clear(name, bit)                                         \
	((name)[_bitstr_byte(bit)] &= ~_bitstr_mask(bit))

/* clear bit N of bitstring name (atomic) */
#define bitstr_clear_atomic(name, bit)                                  \
	(void)os_atomic_andnot(&((name)[_bitstr_byte(bit)]), _bitstr_mask(bit), relaxed)

/* clear bits start ... stop in bitstring */
#define bitstr_nclear(name, start, stop) do {                           \
	bitstr_t *_name = (name);                                       \
	int _start = (start), _stop = (stop);                           \
	int _startbyte = _bitstr_byte(_start);                          \
	int _stopbyte = _bitstr_byte(_stop);                            \
	if (_startbyte == _stopbyte) {                                  \
	        _name[_startbyte] &= ((0xff >> (8 - (_start & 0x7))) |  \
	            (0xff << ((_stop & 0x7) + 1)));                     \
	} else {                                                        \
	        _name[_startbyte] &= 0xff >> (8 - (_start & 0x7));      \
	        while (++_startbyte < _stopbyte)                        \
	                _name[_startbyte] = 0;                          \
	        _name[_stopbyte] &= 0xff << ((_stop & 0x7) + 1);        \
	}                                                               \
} while (0)

/* set bits start ... stop in bitstring */
#define bitstr_nset(name, start, stop) do {                             \
	bitstr_t *_name = (name);                                       \
	int _start = (start), _stop = (stop);                           \
	int _startbyte = _bitstr_byte(_start);                          \
	int _stopbyte = _bitstr_byte(_stop);                            \
	if (_startbyte == _stopbyte) {                                  \
	        _name[_startbyte] |= ((0xff << (_start & 0x7)) &        \
	            (0xff >> (7 - (_stop & 0x7))));                     \
	} else {                                                        \
	        _name[_startbyte] |= 0xff << ((_start) & 0x7);          \
	        while (++_startbyte < _stopbyte)                        \
	                _name[_startbyte] = 0xff;                       \
	        _name[_stopbyte] |= 0xff >> (7 - (_stop & 0x7));        \
	}                                                               \
} while (0)

/* find first bit clear in name */
#define bitstr_ffc(name, nbits, value) do {                             \
	bitstr_t *_name = (name);                                       \
	int _byte, _nbits = (nbits);                                    \
	int _stopbyte = _bitstr_byte(_nbits - 1), _value = -1;          \
	if (_nbits > 0)                                                 \
	        for (_byte = 0; _byte <= _stopbyte; ++_byte)            \
	                if (_name[_byte] != 0xff) {                     \
	                        bitstr_t _lb;                           \
	                        _value = _byte << 3;                    \
	                        for (_lb = _name[_byte]; (_lb & 0x1);   \
	                            ++_value, _lb >>= 1);               \
	                        break;                                  \
	                }                                               \
	if (_value >= nbits)                                            \
	        _value = -1;                                            \
	*(value) = _value;                                              \
} while (0)

/* find first bit set in name */
#define bitstr_ffs(name, nbits, value) do {                             \
	bitstr_t *_name = (name);                                       \
	int _byte, _nbits = (nbits);                                    \
	int _stopbyte = _bitstr_byte(_nbits - 1), _value = -1;          \
	if (_nbits > 0)                                                 \
	        for (_byte = 0; _byte <= _stopbyte; ++_byte)            \
	                if (_name[_byte]) {                             \
	                        bitstr_t _lb;                           \
	                        _value = _byte << 3;                    \
	                        for (_lb = _name[_byte]; !(_lb & 0x1);  \
	                            ++_value, _lb >>= 1);               \
	                        break;                                  \
	                }                                               \
	if (_value >= nbits)                                            \
	        _value = -1;                                            \
	*(value) = _value;                                              \
} while (0)

#endif /* XNU_KERNEL_PRIVATE */
#endif /* !_SYS_BITSTRING_H_ */
