/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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
#include <Kerberos/Kerberos.h>
#include <string.h>


int
main(int argc, char **argv)
{
    krb5_context kcontext = NULL;
    krb5_keytab kt = NULL;
    krb5_keytab_entry entry;
    krb5_kt_cursor cursor = NULL;
    krb5_error_code krb5_err;
    int matched = 0;
    
    char svc_name[] = "host";
    int svc_name_len = (int)strlen (svc_name);
    
    
    krb5_err = krb5_init_context(&kcontext);
    if (krb5_err) { goto Error; }
    krb5_err = krb5_kt_default(kcontext, &kt);
    if (krb5_err) { goto Error; }
    krb5_err = krb5_kt_start_seq_get(kcontext, kt, &cursor);
    if (krb5_err) { goto Error; }
    while ((matched == 0) && (krb5_err = krb5_kt_next_entry(kcontext, kt, &entry, &cursor)) == 0) {
	
	krb5_data *nameData = krb5_princ_name (kcontext, entry.principal);
	if (NULL != nameData->data && svc_name_len == nameData->length
	    && 0 == strncmp (svc_name, nameData->data, nameData->length)) {
	    matched = 1;
	}
        
	krb5_free_keytab_entry_contents(kcontext, &entry);
    }
    
    krb5_err = krb5_kt_end_seq_get(kcontext, kt, &cursor);
 Error:
    if (NULL != kt) { krb5_kt_close (kcontext, kt); }
    if (NULL != kcontext) { krb5_free_context (kcontext); }
    // Return 0 if we got match or -1 if err or no match
    return (0 != krb5_err) ? -1 : matched ? 0 : -1;
}        

