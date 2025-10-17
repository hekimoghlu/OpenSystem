/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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
#include "lutil.h"

/**********************************************************
 *
 * Compare
 *
 **********************************************************/

int
perl_back_compare(
	Operation	*op,
	SlapReply	*rs )
{
	int count;
	char *avastr;

	PerlBackend *perl_back = (PerlBackend *)op->o_bd->be_private;

	avastr = ch_malloc( op->orc_ava->aa_desc->ad_cname.bv_len + 1 +
		op->orc_ava->aa_value.bv_len + 1 );
	
	lutil_strcopy( lutil_strcopy( lutil_strcopy( avastr,
		op->orc_ava->aa_desc->ad_cname.bv_val ), "=" ),
		op->orc_ava->aa_value.bv_val );

	PERL_SET_CONTEXT( PERL_INTERPRETER );
	ldap_pvt_thread_mutex_lock( &perl_interpreter_mutex );	

	{
		dSP; ENTER; SAVETMPS;

		PUSHMARK(sp);
		XPUSHs( perl_back->pb_obj_ref );
		XPUSHs(sv_2mortal(newSVpv( op->o_req_dn.bv_val , 0)));
		XPUSHs(sv_2mortal(newSVpv( avastr , 0)));
		PUTBACK;

		count = call_method("compare", G_SCALAR);

		SPAGAIN;

		if (count != 1) {
			croak("Big trouble in back_compare\n");
		}

		rs->sr_err = POPi;

		PUTBACK; FREETMPS; LEAVE;
	}

	ldap_pvt_thread_mutex_unlock( &perl_interpreter_mutex );	

	ch_free( avastr );

	send_ldap_result( op, rs );

	Debug( LDAP_DEBUG_ANY, "Perl COMPARE\n", 0, 0, 0 );

	return (0);
}

