/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#include <krb5.h>
#include <kcm.h>


static void
add_cred(krb5_context context)
{
    krb5_error_code ret;
    krb5_storage *request, *response;
    krb5_data response_data;
    krb5_data data;

#if 0
    char password[512];

    if (UI_UTIL_read_pw_string(password, sizeof(password),
			       "Password:", 0) != 1)
       errx(1, "failed reading password");
#endif
       
    ret = krb5_kcm_storage_request(context, KCM_OP_ADD_NTLM_CRED, &request);
    if (ret)
	krb5_err(context, 1, ret, "krb5_kcm_storage_request");

    krb5_store_stringz(request, "lha");
    krb5_store_stringz(request, "BUILTIN");
    data.data = "\xac\x8e\x65\x7f\x83\xdf\x82\xbe\xea\x5d\x43\xbd\xaf\x78\x0\xcc"; /* foo */
    data.length = 16;
    krb5_store_data(request, data);

    ret = krb5_kcm_call(context, request, &response, &response_data);
    if (ret)
	krb5_err(context, 1, ret, "krb5_kcm_call");

    krb5_storage_free(request);
    krb5_storage_free(response);
    krb5_data_free(&response_data);
}


static void
list_cred(krb5_context context)
{
    krb5_error_code ret;
    krb5_storage *request, *response;
    krb5_data response_data;
    
    ret = krb5_kcm_storage_request(context, KCM_OP_GET_NTLM_USER_LIST, &request);
    if (ret)
	krb5_err(context, 1, ret, "krb5_kcm_storage_request");

    ret = krb5_kcm_call(context, request, &response, &response_data);
    if (ret)
	krb5_err(context, 1, ret, "krb5_kcm_call");

    while (1) {
	uint32_t morep;
	char *user = NULL, *domain = NULL;

	ret = krb5_ret_uint32(response, &morep);
	if (ret)
	    krb5_err(context, ret, 1, "ret: morep");

	if (morep == 0)
	    break;

	ret = krb5_ret_stringz(response, &user);
	if (ret)
	    krb5_err(context, ret, 1, "ret: user");
	ret = krb5_ret_stringz(response, &domain);
	if (ret)
	    krb5_err(context, ret, 1, "ret: domain");


	printf("user: %s domain: %s\n", user, domain);
    }

    krb5_storage_free(request);
    krb5_storage_free(response);
    krb5_data_free(&response_data);
}



int
main(int argc, char **argv)
{
    krb5_error_code ret;
    krb5_context context;
    
    ret = krb5_init_context(&context);
    if (ret)
	errx(1, "krb5_init_context");

    list_cred(context);

    add_cred(context);

    krb5_free_context(context);

    return 0;
}
