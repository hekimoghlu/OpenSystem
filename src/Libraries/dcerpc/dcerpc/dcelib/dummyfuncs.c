/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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
 */#if defined(__FreeBSD__) && defined(_POSIX_C_SOURCE)
#    undef _POSIX_C_SOURCE
#endif

#include <commonp.h>
#include <stdio.h>
#include <dce/rpc.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>

#ifndef BUILD_RPC_NS_LDAP

#include <ctype.h>

void rpc_ns_binding_export(
	/* [in] */ unsigned32 entry_name_syntax ATTRIBUTE_UNUSED,
	/* [in] */ unsigned_char_p_t entry_name ATTRIBUTE_UNUSED,
	/* [in] */ rpc_if_handle_t if_spec ATTRIBUTE_UNUSED,
	/* [in] */ rpc_binding_vector_p_t binding_vector
	ATTRIBUTE_UNUSED,
	/* [in] */ uuid_vector_p_t object_uuid_vector ATTRIBUTE_UNUSED,
	/* [out] */ unsigned32 *status
)
{
	RPC_DBG_PRINTF(rpc_e_dbg_general, 1, ("rpc_ns_binding_export\n"));
	*status = rpc_s_name_service_unavailable;
}

void
rpc_ns_binding_import_begin(
    /* [in] */ unsigned32 entry_name_syntax ATTRIBUTE_UNUSED,
    /* [in] */ unsigned_char_p_t entry_name ATTRIBUTE_UNUSED,
    /* [in] */ rpc_if_handle_t if_spec ATTRIBUTE_UNUSED,
    /* [in] */ uuid_p_t object_uuid ATTRIBUTE_UNUSED,
    /* [out] */ rpc_ns_handle_t *import_context ATTRIBUTE_UNUSED,
    /* [out] */ unsigned32 *status
)
{
	RPC_DBG_PRINTF(rpc_e_dbg_general, 1, ("rpc_ns_import_begin\n"));
	*status = rpc_s_name_service_unavailable;
}

void
rpc_ns_binding_import_done(
    /* [in, out] */ rpc_ns_handle_t *import_context
	 ATTRIBUTE_UNUSED,
    /* [out] */ unsigned32 *status
)
{
	RPC_DBG_PRINTF(rpc_e_dbg_general, 1, ("rpc_ns_binding_import_done\n"));
	*status = rpc_s_name_service_unavailable;
}

void
rpc_ns_mgmt_handle_set_exp_age(
    /* [in] */ rpc_ns_handle_t ns_handle ATTRIBUTE_UNUSED,
    /* [in] */ unsigned32 expiration_age ATTRIBUTE_UNUSED,
    /* [out] */ unsigned32 *status

)
{
	RPC_DBG_PRINTF(rpc_e_dbg_general, 1,
		("rpc_ns_mgmt_handle_set_exp_age\n"));
	*status = rpc_s_ok;
}

void
rpc_ns_binding_import_next(
    /* [in] */ rpc_ns_handle_t import_context ATTRIBUTE_UNUSED,
    /* [out] */ rpc_binding_handle_t *binding ATTRIBUTE_UNUSED,
    /* [out] */ unsigned32 *status
	       )
{
	RPC_DBG_PRINTF(rpc_e_dbg_general, 1, ("rpc_ns_binding_import_next\n"));
	*status = rpc_s_name_service_unavailable ;
}

#endif /* BUILD_RPC_NS_LDAP */

