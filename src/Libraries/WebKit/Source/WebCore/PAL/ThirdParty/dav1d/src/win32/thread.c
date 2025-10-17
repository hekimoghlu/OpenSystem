/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
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
#include "config.h"

#if defined(_WIN32)

#include <process.h>
#include <stdlib.h>
#include <windows.h>

#include "common/attributes.h"

#include "src/thread.h"

static HRESULT (WINAPI *set_thread_description)(HANDLE, PCWSTR);

COLD void dav1d_init_thread(void) {
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
    HANDLE kernel32 = GetModuleHandleW(L"kernel32.dll");
    if (kernel32)
        set_thread_description =
            (void*)GetProcAddress(kernel32, "SetThreadDescription");
#endif
}

#undef dav1d_set_thread_name
COLD void dav1d_set_thread_name(const wchar_t *const name) {
    if (set_thread_description) /* Only available since Windows 10 1607 */
        set_thread_description(GetCurrentThread(), name);
}

static COLD unsigned __stdcall thread_entrypoint(void *const data) {
    pthread_t *const t = data;
    t->arg = t->func(t->arg);
    return 0;
}

COLD int dav1d_pthread_create(pthread_t *const thread,
                              const pthread_attr_t *const attr,
                              void *(*const func)(void*), void *const arg)
{
    const unsigned stack_size = attr ? attr->stack_size : 0;
    thread->func = func;
    thread->arg = arg;
    thread->h = (HANDLE)_beginthreadex(NULL, stack_size, thread_entrypoint, thread,
                                       STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
    return !thread->h;
}

COLD int dav1d_pthread_join(pthread_t *const thread, void **const res) {
    if (WaitForSingleObject(thread->h, INFINITE))
        return 1;

    if (res)
        *res = thread->arg;

    return !CloseHandle(thread->h);
}

COLD int dav1d_pthread_once(pthread_once_t *const once_control,
                            void (*const init_routine)(void))
{
    BOOL pending = FALSE;

    if (InitOnceBeginInitialize(once_control, 0, &pending, NULL) != TRUE)
        return 1;

    if (pending == TRUE)
        init_routine();

    return !InitOnceComplete(once_control, 0, NULL);
}

#endif
