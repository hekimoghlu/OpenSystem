/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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
/* $Id$ */

#ifndef SPNEGO_LOCL_H
#define SPNEGO_LOCL_H

#include <config.h>

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#include <roken.h>

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#include <gssapi.h>
#include <gssapi_krb5.h>
#include <gssapi_spnego.h>
#include <gssapi_ntlm.h>
#include <gssapi_netlogon.h>
#include <gssapi_spi.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif

#include <heim_threads.h>
#include <asn1_err.h>

#include <gssapi_mech.h>

//XXX #include <heimbase.h>

#include "spnego_asn1.h"
#include "utils.h"
#include <der.h>

#include <heimbase.h>

#define ALLOC(X, N) (X) = calloc((N), sizeof(*(X)))

typedef struct gssspnego_ctx_d *gssspnego_ctx;

typedef OM_uint32
(*gssspnego_initiator_state)(OM_uint32 * minor_status,
			     gss_cred_id_t cred,
			     gssspnego_ctx ctx,
			     gss_name_t name,
			     const gss_OID mech_type,
			     OM_uint32 req_flags,
			     OM_uint32 time_req,
			     const gss_channel_bindings_t input_chan_bindings,
			     const gss_buffer_t input_token,
			     gss_buffer_t output_token,
			     OM_uint32 * ret_flags,
			     OM_uint32 * time_rec);


struct gssspnego_ctx_d {
    gss_buffer_desc	NegTokenInit_mech_types;
    gss_OID		preferred_mech_type;
    gss_OID		selected_mech_type;
    gss_OID		negotiated_mech_type;
    gss_ctx_id_t	negotiated_ctx_id;
    OM_uint32		mech_flags;
    OM_uint32		mech_time_rec;
    gss_name_t		mech_src_name;
    struct spnego_flags {
	unsigned int		open : 1;
	unsigned int		local : 1;
	unsigned int		require_mic : 1;
	unsigned int		peer_require_mic : 1;
    	unsigned int		protocol_require_no_mic : 1;
	unsigned int		sent_mic : 1;
	unsigned int		verified_mic : 1;
	unsigned int		safe_omit : 1;
	unsigned int		maybe_open : 1;
	unsigned int		seen_supported_mech : 1;
    } flags;
    HEIMDAL_MUTEX		ctx_id_mutex;

    gss_name_t		target_name;
    
    gssspnego_initiator_state initiator_state;
};

typedef struct {
	gss_OID_desc		type;
	gss_buffer_desc		value;
	gss_name_t		mech;
} *spnego_name;

extern gss_OID_desc _gss_spnego_mskrb_mechanism_oid_desc;
extern gss_OID_desc _gss_spnego_krb5_mechanism_oid_desc;

#include <spnego-private.h>

#endif /* SPNEGO_LOCL_H */
