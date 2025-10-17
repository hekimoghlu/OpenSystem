/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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

#ifndef _TEST_CCAPI_GLOBALS_H_
#define _TEST_CCAPI_GLOBALS_H_

#include <krb5.h> // gets us TARGET_OS_MAC

#if defined(macintosh) || (defined(__MACH__) && defined(__APPLE__))
#include <TargetConditionals.h>
#endif

#ifdef TARGET_OS_MAC
#include <Kerberos/CredentialsCache.h>
#else
#include <CredentialsCache.h>
#endif

/* GLOBALS */
extern unsigned int total_failure_count;
extern unsigned int failure_count;

extern const char *current_test_name;
extern const char *current_test_activity;

extern const char * ccapi_error_strings[30];

const char *translate_ccapi_error(cc_int32 err);

#define T_CCAPI_INIT \
		do { \
			current_test_name = NULL; \
			current_test_activity = NULL; \
		} while( 0 )

#define BEGIN_TEST(name) \
		do { \
			current_test_name = name; \
			failure_count = 0; \
			test_header(current_test_name);	\
		} while( 0 )

#define BEGIN_CHECK_ONCE(x) \
		do { \
			if (x) { \
				current_test_activity = x; \
			} \
		} while( 0 )

#define END_CHECK_ONCE \
		do { \
			current_test_activity = NULL; \
		} while( 0 )

#define END_TEST_AND_RETURN \
		test_footer(current_test_name, failure_count); \
		total_failure_count += failure_count; \
		return failure_count;

#endif /* _TEST_CCAPI_GLOBALS_H_ */
