/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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
#include "curl_setup.h"

#if defined(USE_MBEDTLS) &&                                     \
  ((defined(USE_THREADS_POSIX) && defined(HAVE_PTHREAD_H)) ||   \
    defined(_WIN32))

#if defined(USE_THREADS_POSIX) && defined(HAVE_PTHREAD_H)
#  include <pthread.h>
#  define MBEDTLS_MUTEX_T pthread_mutex_t
#elif defined(_WIN32)
#  define MBEDTLS_MUTEX_T HANDLE
#endif

#include "mbedtls_threadlock.h"
#include "curl_printf.h"
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

/* number of thread locks */
#define NUMT                    2

/* This array will store all of the mutexes available to Mbedtls. */
static MBEDTLS_MUTEX_T *mutex_buf = NULL;

int Curl_mbedtlsthreadlock_thread_setup(void)
{
  int i;

  mutex_buf = calloc(1, NUMT * sizeof(MBEDTLS_MUTEX_T));
  if(!mutex_buf)
    return 0;     /* error, no number of threads defined */

  for(i = 0;  i < NUMT;  i++) {
#if defined(USE_THREADS_POSIX) && defined(HAVE_PTHREAD_H)
    if(pthread_mutex_init(&mutex_buf[i], NULL))
      return 0; /* pthread_mutex_init failed */
#elif defined(_WIN32)
    mutex_buf[i] = CreateMutex(0, FALSE, 0);
    if(mutex_buf[i] == 0)
      return 0;  /* CreateMutex failed */
#endif /* USE_THREADS_POSIX && HAVE_PTHREAD_H */
  }

  return 1; /* OK */
}

int Curl_mbedtlsthreadlock_thread_cleanup(void)
{
  int i;

  if(!mutex_buf)
    return 0; /* error, no threads locks defined */

  for(i = 0; i < NUMT; i++) {
#if defined(USE_THREADS_POSIX) && defined(HAVE_PTHREAD_H)
    if(pthread_mutex_destroy(&mutex_buf[i]))
      return 0; /* pthread_mutex_destroy failed */
#elif defined(_WIN32)
    if(!CloseHandle(mutex_buf[i]))
      return 0; /* CloseHandle failed */
#endif /* USE_THREADS_POSIX && HAVE_PTHREAD_H */
  }
  free(mutex_buf);
  mutex_buf = NULL;

  return 1; /* OK */
}

int Curl_mbedtlsthreadlock_lock_function(int n)
{
  if(n < NUMT) {
#if defined(USE_THREADS_POSIX) && defined(HAVE_PTHREAD_H)
    if(pthread_mutex_lock(&mutex_buf[n])) {
      DEBUGF(fprintf(stderr,
                     "Error: mbedtlsthreadlock_lock_function failed\n"));
      return 0; /* pthread_mutex_lock failed */
    }
#elif defined(_WIN32)
    if(WaitForSingleObject(mutex_buf[n], INFINITE) == WAIT_FAILED) {
      DEBUGF(fprintf(stderr,
                     "Error: mbedtlsthreadlock_lock_function failed\n"));
      return 0; /* pthread_mutex_lock failed */
    }
#endif /* USE_THREADS_POSIX && HAVE_PTHREAD_H */
  }
  return 1; /* OK */
}

int Curl_mbedtlsthreadlock_unlock_function(int n)
{
  if(n < NUMT) {
#if defined(USE_THREADS_POSIX) && defined(HAVE_PTHREAD_H)
    if(pthread_mutex_unlock(&mutex_buf[n])) {
      DEBUGF(fprintf(stderr,
                     "Error: mbedtlsthreadlock_unlock_function failed\n"));
      return 0; /* pthread_mutex_unlock failed */
    }
#elif defined(_WIN32)
    if(!ReleaseMutex(mutex_buf[n])) {
      DEBUGF(fprintf(stderr,
                     "Error: mbedtlsthreadlock_unlock_function failed\n"));
      return 0; /* pthread_mutex_lock failed */
    }
#endif /* USE_THREADS_POSIX && HAVE_PTHREAD_H */
  }
  return 1; /* OK */
}

#endif /* USE_MBEDTLS */
