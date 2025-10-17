/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#include <config.h>

#include <process.h>

#include <isc/thread.h>
#include <isc/util.h>

isc_result_t
isc_thread_create(isc_threadfunc_t start, isc_threadarg_t arg,
		  isc_thread_t *threadp)
{
	isc_thread_t thread;
	unsigned int id;

	thread = (isc_thread_t)_beginthreadex(NULL, 0, start, arg, 0, &id);
	if (thread == NULL) {
		/* XXX */
		return (ISC_R_UNEXPECTED);
	}

	*threadp = thread;

	return (ISC_R_SUCCESS);
}

isc_result_t
isc_thread_join(isc_thread_t thread, isc_threadresult_t *rp) {
	DWORD result;

	result = WaitForSingleObject(thread, INFINITE);
	if (result != WAIT_OBJECT_0) {
		/* XXX */
		return (ISC_R_UNEXPECTED);
	}
	if (rp != NULL && !GetExitCodeThread(thread, rp)) {
		/* XXX */
		return (ISC_R_UNEXPECTED);
	}
	(void)CloseHandle(thread);

	return (ISC_R_SUCCESS);
}

void
isc_thread_setconcurrency(unsigned int level) {
	/*
	 * This is unnecessary on Win32 systems, but is here so that the
	 * call exists
	 */
}

void
isc_thread_setname(isc_thread_t thread, const char *name) {
	UNUSED(thread);
	UNUSED(name);
}

void *
isc_thread_key_getspecific(isc_thread_key_t key) {
	return(TlsGetValue(key));
}

int
isc_thread_key_setspecific(isc_thread_key_t key, void *value) {
	return (TlsSetValue(key, value) ? 0 : GetLastError());
}

int
isc_thread_key_create(isc_thread_key_t *key, void (*func)(void *)) {
	*key = TlsAlloc();

	return ((*key != -1) ? 0 : GetLastError());
}

int
isc_thread_key_delete(isc_thread_key_t key) {
	return (TlsFree(key) ? 0 : GetLastError());
}
