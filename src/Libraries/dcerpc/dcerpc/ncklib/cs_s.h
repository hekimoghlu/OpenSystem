/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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
**  NAME
**
**      cs_s.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Data types and definitions for Code Set Interoperability extension
**  (or Internationalization) to RPC runtime.
**
**
*/

#if	!defined(_CS_S_H)
#define _CS_S_H

/*
 * ISO 10646:1992, UCS-2, Level 2
 * Universal Code Set (encoding) for DCE
 */
#define UCS2_L2		0x0010101

/*
 * When rpc_ns_mgmt_*** routines are extended in the future to,
 * deal with attributes other than code sets, the new attribute
 * specifier needs to be added here.
 */
#define	RPC_EVAL_TYPE_CODESETS		0x0001
#define	RPC_CUSTOM_EVAL_TYPE_CODESETS	0x0002

/*
 *  Tagged union identifiers.
 *  This is used to distinguish tagged union data structure for
 *  i18n binding handle extension.
 */
#define	RPC_CS_EVAL_TAGS		0
#define RPC_CS_EVAL_METHOD		1

/*
 * Code sets interoperability connection models
 */
#define	RPC_EVAL_NO_CONVERSION		0x0001
#define	RPC_EVAL_RMIR_MODEL		0x0002
#define	RPC_EVAL_CMIR_MODEL		0x0003
#define	RPC_EVAL_SMIR_MODEL		0x0004
#define	RPC_EVAL_INTERMEDIATE_MODEL	0x0005
#define	RPC_EVAL_UNIVERSAL_MODEL	0x0006

/*
 * Extension to an import context handle.  The new field 'eval_routines'
 * in 'rpc_lkup_rep_t' will be the following data structure.
 */
typedef struct {
	unsigned32		num;
	rpc_cs_eval_list_p	list;
} rpc_cs_eval_func_t, *rpc_cs_eval_func_p_t;

/*
 * R P C _ C S _ C O D E S E T _ I 1 4 Y _ D A T A
 *
 * Argument to OSF code set evaluation routine.  This data will be passed
 * to the evaluation routine, and figure out the compatible client and server
 * code sets combination.  The evaluation routine will be called from a client
 * in OSF implementation, and it will not be called from a client stub.
 *
 * ns_name	: NSI entry name for a server
 * cleanup	: boolean flag indicating any clean-up action required.
 * method_p	: pointer to 'rpc_cs_method_eval_t' data.  See above.
 * status	: result of the code set evaluation.
 */
typedef struct codeset_i14y_data {
	unsigned_char_p_t	ns_name;
	void			*args;
	boolean32		cleanup;
	rpc_cs_method_eval_p_t	method_p;
	error_status_t		status;
} rpc_cs_codeset_i14y_data, *rpc_cs_codeset_i14y_data_p;

/*
 * Internal routine to attach the code set interoperability
 * attributes to a binding handle.  This routine is not intended
 * to be used by application developers.  Only runtime uses it.
 */
extern void rpc_cs_binding_set_method (
    /* [in, out] */ rpc_binding_handle_t *h,
    /* [in] */ rpc_cs_method_eval_p_t method_p,
    /* [out] */ error_status_t *status
);

/*
 * prototype declarations for locally defined routines
 */
extern void stub_conversion
    (
	rpc_binding_handle_t	h,
	boolean32		server_side,
	unsigned32		from_tag,
	unsigned32		to_tag,
	byte_t			*conv_ldata,
	unsigned32		conv_l_data_len,
	byte_t			*conv_wdata,
	unsigned32		*conv_p_w_data_len,
	error_status_t		*status
    );

/*
 * Well-known UUID for code set attribute
 */
#define rpc_c_uuid_codesets_string	"a1794860-a955-11cd-8443-08000925d3fe"

extern idl_uuid_t rpc_c_attr_real_codesets;
extern idl_uuid_t *rpc_c_attr_codesets;

#endif	/* !defined(_CS_S_H) */
