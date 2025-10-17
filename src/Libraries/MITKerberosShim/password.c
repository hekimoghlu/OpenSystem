/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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
#include "heim.h"
#include <string.h>

mit_krb5_error_code KRB5_CALLCONV
krb5_set_password_using_ccache(mit_krb5_context context,
			       mit_krb5_ccache ccache,
			       char *newpw,
			       mit_krb5_principal change_password_for,
			       int *result_code,
			       mit_krb5_data *result_code_string,
			       mit_krb5_data *result_string)
{
    krb5_error_code ret;
    krb5_principal target = NULL;
    krb5_data code_string, string;

    LOG_ENTRY();

    if (change_password_for) {
	struct comb_principal *p;
	p = (struct comb_principal *)change_password_for;
	target = p->heim;
    }	

    memset(&code_string, 0, sizeof(code_string));
    memset(&string, 0, sizeof(string));

    ret = heim_krb5_set_password_using_ccache(HC(context),
					      (krb5_ccache)ccache,
					      newpw,
					      target,
					      result_code,
					      &code_string,
					      &string);
    if (ret) {
	LOG_FAILURE(ret, "krb5_set_password_using_ccache");
	return ret;
    }

    if (result_code_string)
	mshim_hdata2mdata(&code_string, result_code_string);
    else
	heim_krb5_data_free(&code_string);

    if (result_string)
	mshim_hdata2mdata(&string, result_string);
    else
	heim_krb5_data_free(&string);
    
    return 0;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_set_password(mit_krb5_context context,
		  mit_krb5_creds *creds,
		  char *newpw,
		  mit_krb5_principal change_password_for,
		  int *result_code,
		  mit_krb5_data *result_code_string,
		  mit_krb5_data *result_string)
{
    krb5_error_code ret;
    krb5_principal target = NULL;
    krb5_data code_string, string;
    krb5_creds hcred;

    LOG_ENTRY();

    if (change_password_for) {
	struct comb_principal *p;
	p = (struct comb_principal *)change_password_for;
	target = p->heim;
    }	

    memset(&code_string, 0, sizeof(code_string));
    memset(&string, 0, sizeof(string));

    mshim_mcred2hcred(HC(context), creds, &hcred);

    ret = heim_krb5_set_password(HC(context),
				 &hcred,
				 newpw,
				 target,
				 result_code,
				 &code_string,
				 &string);
    heim_krb5_free_cred_contents(HC(context), &hcred);
    if (ret) {
	LOG_FAILURE(ret, "krb5_set_password");
	return ret;
    }

    if (result_code_string)
	mshim_hdata2mdata(&code_string, result_code_string);
    else
	heim_krb5_data_free(&code_string);

    if (result_string)
	mshim_hdata2mdata(&string, result_string);
    else
	heim_krb5_data_free(&string);
    
    return 0;
}

