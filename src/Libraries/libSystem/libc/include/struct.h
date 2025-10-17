/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
#ifndef _STRUCT_H_
#define	_STRUCT_H_

/* Offset of the field in the structure. */
#define	fldoff(name, field) \
	((int)&(((struct name *)0)->field))

/* Size of the field in the structure. */
#define	fldsiz(name, field) \
	(sizeof(((struct name *)0)->field))

/* Address of the structure from a field. */
#define	strbase(name, addr, field) \
	((struct name *)((char *)(addr) - fldoff(name, field)))

/*
 * countof() cannot be safely used in a _Static_assert statement, so we provide
 * an unsafe variant that does not verify the input array is statically-defined.
 */
#define countof_unsafe(arr) \
	(sizeof(arr) / sizeof(arr[0]))

/* Number of elements in a statically-defined array */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && __GNUC__
#define countof(arr) ({ \
	_Static_assert( \
			!__builtin_types_compatible_p(typeof(arr), typeof(&(arr)[0])), \
			"array must be statically defined"); \
	(sizeof(arr) / sizeof(arr[0])); \
})
#else
#define countof(arr) \
	countof_unsafe(arr)
#endif

/* Length of a statically-defined string (does not include null terminator) */
#define lenof(str) \
	(sizeof(str) - 1)

/* Last index of a statically-defined array */
#define lastof(arr) \
	(countof(arr) - 1)

#endif /* !_STRUCT_H_ */
