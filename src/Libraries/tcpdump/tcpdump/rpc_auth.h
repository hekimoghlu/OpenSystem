/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
 * auth.h, Authentication interface.
 *
 * Copyright (C) 1984, Sun Microsystems, Inc.
 *
 * The data structures are completely opaque to the client.  The client
 * is required to pass a AUTH * to routines that create rpc
 * "sessions".
 */

/*
 * Status returned from authentication check
 */
enum sunrpc_auth_stat {
	SUNRPC_AUTH_OK=0,
	/*
	 * failed at remote end
	 */
	SUNRPC_AUTH_BADCRED=1,		/* bogus credentials (seal broken) */
	SUNRPC_AUTH_REJECTEDCRED=2,	/* client should begin new session */
	SUNRPC_AUTH_BADVERF=3,		/* bogus verifier (seal broken) */
	SUNRPC_AUTH_REJECTEDVERF=4,	/* verifier expired or was replayed */
	SUNRPC_AUTH_TOOWEAK=5,		/* rejected due to security reasons */
	/*
	 * failed locally
	*/
	SUNRPC_AUTH_INVALIDRESP=6,	/* bogus response verifier */
	SUNRPC_AUTH_FAILED=7		/* some unknown reason */
};

/*
 * Authentication info.  Opaque to client.
 */
struct sunrpc_opaque_auth {
	nd_uint32_t oa_flavor;		/* flavor of auth */
	nd_uint32_t oa_len;		/* length of opaque body */
	/* zero or more bytes of body */
};
