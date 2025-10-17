/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
/* $Id$ */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/*
 * pthread support is disable because the pthread
 * test have no "application pthread libflags" variable,
 * when this is fixed pthread support can be enabled again.
 */
#undef ENABLE_PTHREAD_SUPPORT

#include <sys/param.h>
#ifdef HAVE_SYS_UTSNAME_H
#include <sys/utsname.h>
#endif

#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif

#include <assert.h>
#include <krb5.h>
#include <gssapi/gssapi.h>
#include <gssapi/gssapi_krb5.h>
#include <gssapi/gssapi_spnego.h>
#include <unistd.h>

#include <roken.h>
#include <getarg.h>

#include "protocol.h"

krb5_error_code store_string(krb5_storage *, const char *);


#define ret16(_client, num)					\
    do {							\
        if (krb5_ret_int16((_client)->sock, &(num)) != 0)	\
	    errx(1, "krb5_ret_int16 " #num);		\
    } while(0)

#define ret32(_client, num)					\
    do {							\
        if (krb5_ret_int32((_client)->sock, &(num)) != 0)	\
	    errx(1, "krb5_ret_int32 " #num);		\
    } while(0)

#define retdata(_client, data)					\
    do {							\
        if (krb5_ret_data((_client)->sock, &(data)) != 0)	\
	    errx(1, "krb5_ret_data " #data);		\
    } while(0)

#define retstring(_client, data)					\
    do {							\
        if (krb5_ret_string((_client)->sock, &(data)) != 0)	\
	    errx(1, "krb5_ret_data " #data);		\
    } while(0)


#define put32(_client, num)					\
    do {							\
        if (krb5_store_int32((_client)->sock, num) != 0)	\
	    errx(1, "krb5_store_int32 " #num);	\
    } while(0)

#define putdata(_client, data)					\
    do {							\
        if (krb5_store_data((_client)->sock, data) != 0)	\
	    errx(1, "krb5_store_data " #data);	\
    } while(0)

#define putstring(_client, str)					\
    do {							\
        if (store_string((_client)->sock, str) != 0)		\
	    errx(1, "krb5_store_str " #str);			\
    } while(0)

char *** permutate_all(struct getarg_strings *, size_t *);
