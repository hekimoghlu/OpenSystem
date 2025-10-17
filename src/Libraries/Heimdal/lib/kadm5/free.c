/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#include "kadm5_locl.h"

RCSID("$Id$");

void
kadm5_free_key_data(void *server_handle,
		    int16_t *n_key_data,
		    krb5_key_data *key_data)
{
    int i;
    for(i = 0; i < *n_key_data; i++){
	if(key_data[i].key_data_contents[0]){
	    memset(key_data[i].key_data_contents[0],
		   0,
		   key_data[i].key_data_length[0]);
	    free(key_data[i].key_data_contents[0]);
	}
	if(key_data[i].key_data_contents[1])
	    free(key_data[i].key_data_contents[1]);
    }
    *n_key_data = 0;
}


void
kadm5_free_principal_ent(void *server_handle,
			 kadm5_principal_ent_t princ)
{
    kadm5_server_context *context = server_handle;
    if(princ->principal)
	krb5_free_principal(context->context, princ->principal);
    if(princ->mod_name)
	krb5_free_principal(context->context, princ->mod_name);
    kadm5_free_key_data(server_handle, &princ->n_key_data, princ->key_data);
    while(princ->n_tl_data && princ->tl_data) {
	krb5_tl_data *tp;
	tp = princ->tl_data;
	princ->tl_data = tp->tl_data_next;
	princ->n_tl_data--;
	memset(tp->tl_data_contents, 0, tp->tl_data_length);
	free(tp->tl_data_contents);
	free(tp);
    }
    if (princ->key_data != NULL)
	free (princ->key_data);
    memset(princ, 0, sizeof(*princ));
}

void
kadm5_free_name_list(void *server_handle,
		     char **names,
		     int *count)
{
    int i;
    for(i = 0; i < *count; i++)
	free(names[i]);
    free(names);
    *count = 0;
}
