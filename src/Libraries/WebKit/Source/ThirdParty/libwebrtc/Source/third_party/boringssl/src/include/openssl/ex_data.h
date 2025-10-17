/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
/* ====================================================================
 * Copyright (c) 1998-2001 The OpenSSL Project.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. All advertising materials mentioning features or use of this
 *    software must display the following acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit. (http://www.openssl.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@openssl.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.openssl.org/)"
 *
 * THIS SOFTWARE IS PROVIDED BY THE OpenSSL PROJECT ``AS IS'' AND ANY
 * EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE OpenSSL PROJECT OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 *
 * This product includes cryptographic software written by Eric Young
 * (eay@cryptsoft.com).  This product includes software written by Tim
 * Hudson (tjh@cryptsoft.com). */

#ifndef OPENSSL_HEADER_EX_DATA_H
#define OPENSSL_HEADER_EX_DATA_H

#include <openssl/base.h>

#include <openssl/stack.h>

#if defined(__cplusplus)
extern "C" {
#endif


// ex_data is a mechanism for associating arbitrary extra data with objects.
// For each type of object that supports ex_data, different users can be
// assigned indexes in which to store their data. Each index has callback
// functions that are called when an object of that type is freed or
// duplicated.


typedef struct crypto_ex_data_st CRYPTO_EX_DATA;


// Type-specific functions.

#if 0  // Sample

// Each type that supports ex_data provides three functions:

// TYPE_get_ex_new_index allocates a new index for |TYPE|. An optional
// |free_func| argument may be provided which is called when the owning object
// is destroyed. See |CRYPTO_EX_free| for details. The |argl| and |argp|
// arguments are opaque values that are passed to the callback. It returns the
// new index or a negative number on error.
OPENSSL_EXPORT int TYPE_get_ex_new_index(long argl, void *argp,
                                         CRYPTO_EX_unused *unused,
                                         CRYPTO_EX_dup *dup_unused,
                                         CRYPTO_EX_free *free_func);

// TYPE_set_ex_data sets an extra data pointer on |t|. The |index| argument
// must have been returned from a previous call to |TYPE_get_ex_new_index|.
OPENSSL_EXPORT int TYPE_set_ex_data(TYPE *t, int index, void *arg);

// TYPE_get_ex_data returns an extra data pointer for |t|, or NULL if no such
// pointer exists. The |index| argument should have been returned from a
// previous call to |TYPE_get_ex_new_index|.
OPENSSL_EXPORT void *TYPE_get_ex_data(const TYPE *t, int index);

// Some types additionally preallocate index zero, with all callbacks set to
// NULL. Applications that do not need the general ex_data machinery may use
// this instead.

// TYPE_set_app_data sets |t|'s application data pointer to |arg|. It returns
// one on success and zero on error.
OPENSSL_EXPORT int TYPE_set_app_data(TYPE *t, void *arg);

// TYPE_get_app_data returns the application data pointer for |t|, or NULL if no
// such pointer exists.
OPENSSL_EXPORT void *TYPE_get_app_data(const TYPE *t);

#endif  // Sample


// Callback types.

// CRYPTO_EX_free is a callback function that is called when an object of the
// class with extra data pointers is being destroyed. For example, if this
// callback has been passed to |SSL_get_ex_new_index| then it may be called each
// time an |SSL*| is destroyed.
//
// The callback is passed the to-be-destroyed object (i.e. the |SSL*|) in
// |parent|. As |parent| will shortly be destroyed, callers must not perform
// operations that would increment its reference count, pass ownership, or
// assume the object outlives the function call. The arguments |argl| and |argp|
// contain opaque values that were given to |CRYPTO_get_ex_new_index_ex|.
//
// This callback may be called with a NULL value for |ptr| if |parent| has no
// value set for this index. However, the callbacks may also be skipped entirely
// if no extra data pointers are set on |parent| at all.
typedef void CRYPTO_EX_free(void *parent, void *ptr, CRYPTO_EX_DATA *ad,
                            int index, long argl, void *argp);


// Deprecated functions.

// CRYPTO_cleanup_all_ex_data does nothing.
OPENSSL_EXPORT void CRYPTO_cleanup_all_ex_data(void);

// CRYPTO_EX_dup is a legacy callback function type which is ignored.
typedef int CRYPTO_EX_dup(CRYPTO_EX_DATA *to, const CRYPTO_EX_DATA *from,
                          void **from_d, int index, long argl, void *argp);


// Private structures.

// CRYPTO_EX_unused is a placeholder for an unused callback. It is aliased to
// int to ensure non-NULL callers fail to compile rather than fail silently.
typedef int CRYPTO_EX_unused;

struct crypto_ex_data_st {
  STACK_OF(void) *sk;
};


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_EX_DATA_H
