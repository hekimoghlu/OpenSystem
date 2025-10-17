/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 1999-2011 The OpenLDAP Foundation.
 * Portions Copyright 1999 John C. Quillan.
 * Portions Copyright 2002 myinternet Limited.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted only as authorized by the OpenLDAP
 * Public License.
 *
 * A copy of this license is available in file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */

#include "perl_back.h"
#include "../config.h"
/**********************************************************
 *
 * Close
 *
 **********************************************************/

int
perl_back_close(
	BackendInfo *bd
)
{
	perl_destruct(PERL_INTERPRETER);
	perl_free(PERL_INTERPRETER);
	PERL_INTERPRETER = NULL;
#ifdef PERL_SYS_TERM
	PERL_SYS_TERM();
#endif

	ldap_pvt_thread_mutex_destroy( &perl_interpreter_mutex );	

	return 0;
}

int
perl_back_db_destroy(
	BackendDB *be,
	ConfigReply *cr
)
{
	PerlBackend *pb = be->be_private;

	ch_free( pb->pb_module_name );
	ber_bvarray_free( pb->pb_module_path );
	ber_bvarray_free( pb->pb_module_config );

	free( be->be_private );
	be->be_private = NULL;

	return 0;
}
