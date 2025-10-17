/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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

#include <stdio.h>
#include <limits.h>

#include <Kerberos/Kerberos.h>

#include "test_ccapi_check.h"
#include "test_ccapi_constants.h"
#include "test_ccapi_context.h"
#include "test_ccapi_ccache.h"
#include "test_ccapi_iterators.h"
#include "test_ccapi_v2.h"

int main (int argc, const char * argv[]) {
    
	cc_int32 err = ccNoError;
//	cc_ccache_iterator_t cache_iterator = NULL;
//	cc_credentials_iterator_t cred_iterator = NULL;
	
	fprintf(stdout, "Testing CCAPI against CCAPI v3 rev 8 documentation...\n");
	fprintf(stdout, "Warning: this test suite is woefully incomplete and unpolished.\n");
	
	T_CCAPI_INIT;
#if 0	
	// *** ccapi v2 compat ***
	err = check_cc_shutdown();
	err = check_cc_get_change_time();
	err = check_cc_open();
	err = check_cc_create();
	err = check_cc_close();
	err = check_cc_destroy();
	err = check_cc_get_cred_version();
	err = check_cc_get_name();
	err = check_cc_get_principal();
	err = check_cc_set_principal();
	err = check_cc_store();
	err = check_cc_remove_cred();
	err = check_cc_seq_fetch_NCs_begin();
	err = check_cc_seq_fetch_NCs_next();
	err = check_cc_seq_fetch_creds_begin();
	err = check_cc_seq_fetch_creds_next();
	err = check_cc_get_NC_info();
#endif
	err = check_constants();
	
	// *** cc_context ***
	err = check_cc_initialize();
	err = check_cc_context_release();
	err = check_cc_context_get_change_time();
	err = check_cc_context_get_default_ccache_name();
	err = check_cc_context_open_ccache();
	err = check_cc_context_open_default_ccache();
	err = check_cc_context_create_ccache();
	err = check_cc_context_create_default_ccache();
	err = check_cc_context_create_new_ccache();
	err = check_cc_context_new_ccache_iterator();
	// err = check_cc_context_lock();
	// err = check_cc_context_unlock();
	err = check_cc_context_compare();
	
	// *** cc_ccache ***
	err = check_cc_ccache_release();
	err = check_cc_ccache_destroy();
	err = check_cc_ccache_set_default();
	err = check_cc_ccache_get_credentials_version();
	err = check_cc_ccache_get_name();
	err = check_cc_ccache_get_principal();
	err = check_cc_ccache_set_principal();
	err = check_cc_ccache_store_credentials();
	err = check_cc_ccache_remove_credentials();
	err = check_cc_ccache_new_credentials_iterator();
	// err = check_cc_ccache_lock();
	// err = check_cc_ccache_unlock();
	err = check_cc_ccache_get_change_time();
	err = check_cc_ccache_get_last_default_time();
	err = check_cc_ccache_move();
	err = check_cc_ccache_compare();
	err = check_cc_ccache_get_kdc_time_offset();
	err = check_cc_ccache_set_kdc_time_offset();
	err = check_cc_ccache_clear_kdc_time_offset();
	
	// *** cc_ccache_iterator ***
	err = check_cc_ccache_iterator_next();
	
	// *** cc_credentials_iterator ***
	err = check_cc_credentials_iterator_next();
	
	fprintf(stdout, "\nFinished testing CCAPI. %d failure%s in total.\n", total_failure_count, (total_failure_count == 1) ? "" : "s");
	
    return err;
}
