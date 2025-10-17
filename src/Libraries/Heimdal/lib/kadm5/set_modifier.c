/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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

kadm5_ret_t
_kadm5_set_modifier(kadm5_server_context *context,
		    hdb_entry *ent)
{
    kadm5_ret_t ret;
    if(ent->modified_by == NULL){
	ent->modified_by = malloc(sizeof(*ent->modified_by));
	if(ent->modified_by == NULL)
	    return ENOMEM;
    } else
	free_Event(ent->modified_by);
    ent->modified_by->time = time(NULL);
    ret = krb5_copy_principal(context->context, context->caller,
			      &ent->modified_by->principal);
    return ret;
}

