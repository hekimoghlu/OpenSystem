/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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
**     cs_s_conv.c
**
**  FACILITY:
**
**     Remote Procedure Call (RPC)
**     I18N Character Set Conversion Call   (RPC)
**
**  ABSTRACT:
**
**
*/
#include <commonp.h>		/* include nbase.h lbase.h internally	*/
#include <com.h>		/* definition of rpc_binding_rep_p_t	*/
#include <dce/rpcsts.h>
#include <codesets.h>		/* Data definitions for I18N NSI
							sub-component   */
#include <stdio.h>		/* definition of NULL			*/
#include <stdlib.h>		/* definition of MB_CUR_MAX		*/
#include <iconv.h>		/* definition of iconv routines		*/
#include <langinfo.h>		/* definition of nl_langinfo routine	*/
#include <string.h>		/* definition of strncpy routine	*/
#include <errno.h>		/* definition of error numbers 		*/

#include <codesets.h>
#include <cs_s.h>		/* Private defs for code set interoperability */


void stub_conversion
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
)
{
	iconv_t			cd;
	byte_t			*ldata = conv_ldata;
	byte_t			*wdata = conv_wdata;
	int			size;
	int			inbytesleft;
	int			outbytesleft;
	char			*iconv_from_cd;
	char			*iconv_to_cd;
	int			i_ret;
	int			init_len;

	dce_cs_rgy_to_loc (
		from_tag,
		(idl_char **)&iconv_from_cd,
		NULL,
		NULL,
		status );

	if (*status != dce_cs_c_ok)
		return;

	dce_cs_rgy_to_loc (
		to_tag,
		(idl_char **)&iconv_to_cd,
		NULL,
		NULL,
		status );

	if (*status != dce_cs_c_ok)
		return;

	if ((cd = iconv_open(iconv_to_cd, iconv_from_cd)) == (iconv_t)-1)
	{
		*status = rpc_s_ss_incompatible_codesets;
		return;
	}

	/* Set the number of bytes left in input buffer */
	init_len = strlen((char *)ldata);
	inbytesleft = init_len;
	outbytesleft = (int)conv_l_data_len * sizeof(unsigned_char_t);

	i_ret = iconv(cd, (char **)&ldata, &inbytesleft, (char **)&wdata, &outbytesleft);

	if (i_ret)	/* Iconv returns zero when it succeed */
	{
		if (errno == EILSEQ)
			*status = rpc_s_ss_invalid_char_input;
		else if (errno = E2BIG)
			*status = rpc_s_ss_short_conv_buffer;
		else if (errno = EINVAL)
			*status = rpc_s_ss_invalid_char_input;
		i_ret = iconv_close(cd);
		return;
	}
	*wdata = '\0';	/* Guard against a stale data */

	if ((i_ret = iconv_close(cd)) == -1)
	{
		*status = rpc_s_ss_iconv_error;
		return;
	}

	if (conv_p_w_data_len != NULL)
	{
		*conv_p_w_data_len = strlen((char *)conv_wdata);
	}

	*status = rpc_s_ok;
	return;
}
