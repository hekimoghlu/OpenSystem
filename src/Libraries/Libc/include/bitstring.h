/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#ifndef _BITSTRING_H_
#define	_BITSTRING_H_

#include <_bounds.h>
#include <stdlib.h>

_LIBC_SINGLE_BY_DEFAULT()

typedef	unsigned char bitstr_t;

/* internal macros */
				/* byte of the bitstring bit is in */
#define	_bit_byte(bit) \
	((bit) >> 3)

				/* mask for the bit within its byte */
#define	_bit_mask(bit) \
	(1 << ((bit)&0x7))

/* external macros */
				/* bytes in a bitstring of nbits bits */
#define	bitstr_size(nbits) \
	(((nbits) + 7) >> 3)

				/* allocate a bitstring */
#define	bit_alloc(nbits) \
	(bitstr_t *)calloc((size_t)bitstr_size(nbits), sizeof(bitstr_t))

				/* allocate a bitstring on the stack */
#define	bit_decl(name, nbits) \
	((name)[bitstr_size(nbits)])

				/* is bit N of bitstring name set? */
#define	bit_test(name, bit) \
	((name)[_bit_byte(bit)] & _bit_mask(bit))

				/* set bit N of bitstring name */
#define	bit_set(name, bit) \
	((name)[_bit_byte(bit)] |= _bit_mask(bit))

				/* clear bit N of bitstring name */
#define	bit_clear(name, bit) \
	((name)[_bit_byte(bit)] &= ~_bit_mask(bit))

				/* clear bits start ... stop in bitstring */
#define	bit_nclear(name, start, stop) do { \
	bitstr_t *_name = (name); \
	int _start = (start), _stop = (stop); \
	int _startbyte = _bit_byte(_start); \
	int _stopbyte = _bit_byte(_stop); \
	if (_startbyte == _stopbyte) { \
		_name[_startbyte] &= ((0xff >> (8 - (_start&0x7))) | \
				      (0xff << ((_stop&0x7) + 1))); \
	} else { \
		_name[_startbyte] &= 0xff >> (8 - (_start&0x7)); \
		while (++_startbyte < _stopbyte) \
			_name[_startbyte] = 0; \
		_name[_stopbyte] &= 0xff << ((_stop&0x7) + 1); \
	} \
} while (0)

				/* set bits start ... stop in bitstring */
#define	bit_nset(name, start, stop) do { \
	bitstr_t *_name = (name); \
	int _start = (start), _stop = (stop); \
	int _startbyte = _bit_byte(_start); \
	int _stopbyte = _bit_byte(_stop); \
	if (_startbyte == _stopbyte) { \
		_name[_startbyte] |= ((0xff << (_start&0x7)) & \
				    (0xff >> (7 - (_stop&0x7)))); \
	} else { \
		_name[_startbyte] |= 0xff << ((_start)&0x7); \
		while (++_startbyte < _stopbyte) \
	    		_name[_startbyte] = 0xff; \
		_name[_stopbyte] |= 0xff >> (7 - (_stop&0x7)); \
	} \
} while (0)

				/* find first bit clear in name */
#define	bit_ffc(name, nbits, value) do { \
	bitstr_t *_name = (name); \
	int _byte, _nbits = (nbits); \
	int _stopbyte = _bit_byte(_nbits - 1), _value = -1; \
	if (_nbits > 0) \
		for (_byte = 0; _byte <= _stopbyte; ++_byte) \
			if (_name[_byte] != 0xff) { \
				bitstr_t _lb; \
				_value = _byte << 3; \
				for (_lb = _name[_byte]; (_lb&0x1); \
				    ++_value, _lb >>= 1); \
				break; \
			} \
	if (_value >= nbits) \
		_value = -1; \
	*(value) = _value; \
} while (0)

				/* find first bit set in name */
#define	bit_ffs(name, nbits, value) do { \
	bitstr_t *_name = (name); \
	int _byte, _nbits = (nbits); \
	int _stopbyte = _bit_byte(_nbits - 1), _value = -1; \
	if (_nbits > 0) \
		for (_byte = 0; _byte <= _stopbyte; ++_byte) \
			if (_name[_byte]) { \
				bitstr_t _lb; \
				_value = _byte << 3; \
				for (_lb = _name[_byte]; !(_lb&0x1); \
				    ++_value, _lb >>= 1); \
				break; \
			} \
	if (_value >= nbits) \
		_value = -1; \
	*(value) = _value; \
} while (0)

#endif /* !_BITSTRING_H_ */
