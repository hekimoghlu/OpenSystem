/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 7, 2022.
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
#include <err.h>
#include <string.h>
#include <Kerberos/krb5.h>
#include "test_collection.h"

int main(int argc, char **argv)
{
	krb5_error_code ret = 0;

	krb5_context context = NULL;
	krb5_principal princ = NULL;
	krb5_get_init_creds_opt opt;
	krb5_creds cred;
	int result_code;
	krb5_data result_code_string;
	krb5_data result_string;

	test_collection_t *tc = NULL;

	memset(&cred, 0, sizeof(cred));
	memset(&result_code_string, 0, sizeof(result_code_string));
	memset(&result_string, 0, sizeof(result_string));

	tc = tests_init_and_start("test-krb5");
	tests_set_flags(tc, TC_FLAG_EXIT_ON_FAILURE);
	tests_set_total_count_hint(tc, 5);

	ret = krb5_init_context(&context);
	test_evaluate(tc, "krb5_init_context", ret);

	ret = krb5_parse_name(context, argv[1], &princ);
	test_evaluate(tc, "krb5_parse_name", ret);

	krb5_get_init_creds_opt_init(&opt);

	ret = krb5_get_init_creds_password (context,
					    &cred,
					    princ,
					    argv[2],
					    krb5_prompter_posix,
					    NULL,
					    0,
					    "kadmin/changepw",
					    &opt);
	test_evaluate(tc, "krb5_get_init_creds_password", ret);

	ret = krb5_set_password(context,
				&cred,
				argv[3],
				NULL,
				&result_code,
				&result_code_string,
				&result_string);
	test_evaluate(tc, "krb5_set_password", ret);

	printf("result code: %d result_code_string %.*s result_string: %*.s\n",
	       result_code,
	       (int)result_code_string.length,
	       (char *)result_code_string.data,
	       (int)result_string.length,
	       (char *)result_string.data);

	test_evaluate(tc, "krb5_set_password result code", result_code);

	krb5_free_context(context);

	return tests_stop_and_free(tc);
}
