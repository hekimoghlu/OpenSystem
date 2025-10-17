/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
/* theller: almost verbatim translation from gas syntax to MSVC inline
   assembler code. */

/* theller: ffi_call_x86 now returns an integer - the difference of the stack
   pointer before and after the function call.  If everything is ok, zero is
   returned.  If stdcall functions are passed the wrong number of arguments,
   the difference will be nonzero. */

#include <ffi.h>
#include <ffi_common.h>

__declspec(naked) int
ffi_call_x86(void (* prepfunc)(char *, extended_cif *), /* 8 */
	     extended_cif *ecif, /* 12 */
	     unsigned bytes, /* 16 */
	     unsigned flags, /* 20 */
	     unsigned *rvalue, /* 24 */
	     void (*fn)()) /* 28 */
{
	_asm {
		push ebp
		mov ebp, esp

		push esi // NEW: this register must be preserved across function calls
// XXX SAVE ESP NOW!
		mov esi, esp		// save stack pointer before the call

// Make room for all of the new args.
		mov ecx, [ebp+16]
		sub esp, ecx		// sub esp, bytes
		
		mov eax, esp

// Place all of the ffi_prep_args in position
		push [ebp + 12] // ecif
		push eax
		call [ebp + 8] // prepfunc

// Return stack to previous state and call the function
		add esp, 8
// FIXME: Align the stack to a 128-bit boundary to avoid
// potential performance hits.
		call [ebp + 28]

// Load ecif->cif->abi
		mov ecx, [ebp + 12]
		mov ecx, [ecx]ecif.cif
		mov ecx, [ecx]ecif.cif.abi
		
		cmp ecx, FFI_STDCALL
		je noclean
// STDCALL: Remove the space we pushed for the args
		mov ecx, [ebp + 16]
		add esp, ecx
// CDECL: Caller has already cleaned the stack
noclean:
// Check that esp has the same value as before!
		sub esi, esp

// Load %ecx with the return type code
		mov ecx, [ebp + 20]

// If the return value pointer is NULL, assume no return value.
/*
  Intel asm is weird. We have to explicitely specify 'DWORD PTR' in the nexr instruction,
  otherwise only one BYTE will be compared (instead of a DWORD)!
 */
		cmp DWORD PTR [ebp + 24], 0
		jne sc_retint

// Even if there is no space for the return value, we are
// obliged to handle floating-point values.
		cmp ecx, FFI_TYPE_FLOAT
		jne sc_noretval
//        fstp  %st(0)
		fstp st(0)

		jmp sc_epilogue

sc_retint:
		cmp ecx, FFI_TYPE_INT
		jne sc_retfloat
//        # Load %ecx with the pointer to storage for the return value
		mov ecx, [ebp + 24]
		mov [ecx + 0], eax
		jmp sc_epilogue

sc_retfloat:
		cmp ecx, FFI_TYPE_FLOAT
		jne sc_retdouble
// Load %ecx with the pointer to storage for the return value
		mov ecx, [ebp+24]
//        fstps (%ecx)
		fstp DWORD PTR [ecx]
		jmp sc_epilogue

sc_retdouble:
		cmp ecx, FFI_TYPE_DOUBLE
		jne sc_retlongdouble
//        movl  24(%ebp),%ecx
		mov ecx, [ebp+24]
		fstp QWORD PTR [ecx]
		jmp sc_epilogue

		jmp sc_retlongdouble // avoid warning about unused label
sc_retlongdouble:
		cmp ecx, FFI_TYPE_LONGDOUBLE
		jne sc_retint64
// Load %ecx with the pointer to storage for the return value
		mov ecx, [ebp+24]
//        fstpt (%ecx)
		fstp QWORD PTR [ecx] /* XXX ??? */
		jmp sc_epilogue

sc_retint64:
		cmp ecx, FFI_TYPE_SINT64
		jne sc_retstruct
// Load %ecx with the pointer to storage for the return value
		mov ecx, [ebp+24]
		mov [ecx+0], eax
		mov [ecx+4], edx

sc_retstruct:
// Nothing to do!

sc_noretval:
sc_epilogue:
		mov eax, esi
		pop esi // NEW restore: must be preserved across function calls
		mov esp, ebp
		pop ebp
		ret
	}
}
