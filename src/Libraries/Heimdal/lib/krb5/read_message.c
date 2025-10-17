/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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
#include "krb5_locl.h"

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_read_message (krb5_context context,
		   krb5_pointer p_fd,
		   krb5_data *data)
{
    krb5_error_code ret;
    krb5_ssize_t sret;
    uint32_t len;
    uint8_t buf[4];

    krb5_data_zero(data);

    sret = krb5_net_read (context, p_fd, buf, 4);
    if(sret == -1) {
	ret = errno;
	krb5_clear_error_message (context);
	return ret;
    }
    if(sret < 4) {
	krb5_clear_error_message(context);
	return HEIM_ERR_EOF;
    }
    len = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
    if (len > UINT_MAX / 16) {
	krb5_set_error_message(context, ERANGE, N_("packet to large", ""));
	return ERANGE;
    }
    ret = krb5_data_alloc (data, len);
    if (ret) {
	krb5_clear_error_message(context);
	return ret;
    }
    if (krb5_net_read (context, p_fd, data->data, len) != (ssize_t)len) {
	ret = errno;
	krb5_data_free (data);
	krb5_clear_error_message (context);
	return ret;
    }
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_read_priv_message(krb5_context context,
		       krb5_auth_context ac,
		       krb5_pointer p_fd,
		       krb5_data *data)
{
    krb5_error_code ret;
    krb5_data packet;

    ret = krb5_read_message(context, p_fd, &packet);
    if(ret)
	return ret;
    ret = krb5_rd_priv (context, ac, &packet, data, NULL);
    krb5_data_free(&packet);
    return ret;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_read_safe_message(krb5_context context,
		       krb5_auth_context ac,
		       krb5_pointer p_fd,
		       krb5_data *data)
{
    krb5_error_code ret;
    krb5_data packet;

    ret = krb5_read_message(context, p_fd, &packet);
    if(ret)
	return ret;
    ret = krb5_rd_safe (context, ac, &packet, data, NULL);
    krb5_data_free(&packet);
    return ret;
}
