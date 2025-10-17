/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#define L(x) x

#define SF_ARG9		MODE_CHOICE(56,112)
#define SF_ARG10	MODE_CHOICE(60,120)
#define SF_ARG11	MODE_CHOICE(64,128)
#define SF_ARG12	MODE_CHOICE(68,136)
#define SF_ARG13	MODE_CHOICE(72,144)
#define SF_ARG14	MODE_CHOICE(76,152)
#define SF_ARG15	MODE_CHOICE(80,160)
#define SF_ARG16	MODE_CHOICE(84,168)
#define SF_ARG17	MODE_CHOICE(88,176)
#define SF_ARG18	MODE_CHOICE(92,184)
#define SF_ARG19	MODE_CHOICE(96,192)
#define SF_ARG20	MODE_CHOICE(100,200)
#define SF_ARG21	MODE_CHOICE(104,208)
#define SF_ARG22	MODE_CHOICE(108,216)
#define SF_ARG23	MODE_CHOICE(112,224)
#define SF_ARG24	MODE_CHOICE(116,232)
#define SF_ARG25	MODE_CHOICE(120,240)
#define SF_ARG26	MODE_CHOICE(124,248)
#define SF_ARG27	MODE_CHOICE(128,256)
#define SF_ARG28	MODE_CHOICE(132,264)
#define SF_ARG29	MODE_CHOICE(136,272)

#define ASM_NEEDS_REGISTERS		4
#define NUM_GPR_ARG_REGISTERS	8
#define NUM_FPR_ARG_REGISTERS	13

#define FFI_TYPE_1_BYTE(x)	((x) == FFI_TYPE_UINT8 || (x) == FFI_TYPE_SINT8)
#define FFI_TYPE_2_BYTE(x)	((x) == FFI_TYPE_UINT16 || (x) == FFI_TYPE_SINT16)
#define FFI_TYPE_4_BYTE(x)	\
	((x) == FFI_TYPE_UINT32 || (x) == FFI_TYPE_SINT32 ||\
	(x) == FFI_TYPE_INT || (x) == FFI_TYPE_FLOAT)

#if !defined(LIBFFI_ASM)

enum {
	FLAG_RETURNS_NOTHING	= 1 << (31 - 30),	//	cr7
	FLAG_RETURNS_FP			= 1 << (31 - 29),
	FLAG_RETURNS_64BITS		= 1 << (31 - 28),
	FLAG_RETURNS_128BITS	= 1 << (31 - 31),

	FLAG_RETURNS_STRUCT		= 1 << (31 - 27),	//	cr6
	FLAG_STRUCT_CONTAINS_FP	= 1 << (31 - 26),

	FLAG_ARG_NEEDS_COPY		= 1 << (31 - 7),
	FLAG_FP_ARGUMENTS		= 1 << (31 - 6),	//	cr1.eq; specified by ABI
	FLAG_4_GPR_ARGUMENTS	= 1 << (31 - 5),
	FLAG_RETVAL_REFERENCE	= 1 << (31 - 4)
};

#if defined(__ppc64__)
void ffi64_struct_to_ram_form(const ffi_type*, const char*, unsigned int*,
	const char*, unsigned int*, unsigned int*, char*, unsigned int*);
void ffi64_struct_to_reg_form(const ffi_type*, const char*, unsigned int*,
	unsigned int*, char*, unsigned int*, char*, unsigned int*);
bool ffi64_stret_needs_ptr(const ffi_type* inType,
	unsigned short*, unsigned short*);
#endif

#endif	// !defined(LIBFFI_ASM)