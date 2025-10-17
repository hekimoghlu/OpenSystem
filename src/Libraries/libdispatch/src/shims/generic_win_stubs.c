/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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

#include "internal.h"

typedef void (WINAPI *_precise_time_fn_t)(PULONGLONG);

DISPATCH_STATIC_GLOBAL(dispatch_once_t _dispatch_precise_time_pred);
DISPATCH_STATIC_GLOBAL(_precise_time_fn_t _dispatch_QueryInterruptTimePrecise_ptr);
DISPATCH_STATIC_GLOBAL(_precise_time_fn_t _dispatch_QueryUnbiasedInterruptTimePrecise_ptr);

typedef NTSTATUS (NTAPI *_NtQueryInformationFile_fn_t)(HANDLE FileHandle,
		PIO_STATUS_BLOCK IoStatusBlock, PVOID FileInformation, ULONG Length,
		FILE_INFORMATION_CLASS FileInformationClass);

DISPATCH_STATIC_GLOBAL(dispatch_once_t _dispatch_ntdll_pred);
DISPATCH_STATIC_GLOBAL(_NtQueryInformationFile_fn_t _dispatch_NtQueryInformationFile_ptr);

bool
_dispatch_handle_is_socket(HANDLE hFile)
{
	// GetFileType() returns FILE_TYPE_PIPE for both pipes and sockets. We can
	// disambiguate by checking if PeekNamedPipe() fails with
	// ERROR_INVALID_FUNCTION.
	if (GetFileType(hFile) == FILE_TYPE_PIPE &&
			!PeekNamedPipe(hFile, NULL, 0, NULL, NULL, NULL)) {
		return GetLastError() == ERROR_INVALID_FUNCTION;
	}
	return false;
}

static void
_dispatch_init_precise_time(void *context DISPATCH_UNUSED)
{
	HMODULE kernelbase = LoadLibraryW(L"KernelBase.dll");
	if (!kernelbase) {
		DISPATCH_INTERNAL_CRASH(0, "failed to load KernelBase.dll");
	}
	_dispatch_QueryInterruptTimePrecise_ptr = (_precise_time_fn_t)
			GetProcAddress(kernelbase, "QueryInterruptTimePrecise");
	_dispatch_QueryUnbiasedInterruptTimePrecise_ptr = (_precise_time_fn_t)
			GetProcAddress(kernelbase, "QueryUnbiasedInterruptTimePrecise");
	if (!_dispatch_QueryInterruptTimePrecise_ptr) {
		DISPATCH_INTERNAL_CRASH(0, "could not locate QueryInterruptTimePrecise");
	}
	if (!_dispatch_QueryUnbiasedInterruptTimePrecise_ptr) {
		DISPATCH_INTERNAL_CRASH(0, "could not locate QueryUnbiasedInterruptTimePrecise");
	}
}

void
_dispatch_QueryInterruptTimePrecise(PULONGLONG lpInterruptTimePrecise)
{
	dispatch_once_f(&_dispatch_precise_time_pred, NULL, _dispatch_init_precise_time);
	return _dispatch_QueryInterruptTimePrecise_ptr(lpInterruptTimePrecise);
}

void
_dispatch_QueryUnbiasedInterruptTimePrecise(PULONGLONG lpUnbiasedInterruptTimePrecise)
{
	dispatch_once_f(&_dispatch_precise_time_pred, NULL, _dispatch_init_precise_time);
	return _dispatch_QueryUnbiasedInterruptTimePrecise_ptr(lpUnbiasedInterruptTimePrecise);
}

static void
_dispatch_init_ntdll(void *context DISPATCH_UNUSED)
{
	HMODULE ntdll = LoadLibraryW(L"ntdll.dll");
	if (!ntdll) {
		// ntdll is not required.
		return;
	}
	_dispatch_NtQueryInformationFile_ptr = (_NtQueryInformationFile_fn_t)
			GetProcAddress(ntdll, "NtQueryInformationFile");
}

NTSTATUS _dispatch_NtQueryInformationFile(HANDLE FileHandle,
		PIO_STATUS_BLOCK IoStatusBlock, PVOID FileInformation, ULONG Length,
		FILE_INFORMATION_CLASS FileInformationClass)
{
	dispatch_once_f(&_dispatch_ntdll_pred, NULL, _dispatch_init_ntdll);
	if (!_dispatch_NtQueryInformationFile_ptr) {
		return STATUS_NOT_SUPPORTED;
	}
	return _dispatch_NtQueryInformationFile_ptr(FileHandle, IoStatusBlock,
			FileInformation, Length, FileInformationClass);
}
