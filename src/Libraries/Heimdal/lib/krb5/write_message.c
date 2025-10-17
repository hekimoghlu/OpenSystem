/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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
krb5_write_message (krb5_context context,
		    krb5_pointer p_fd,
		    krb5_data *data)
{
    uint32_t len;
    uint8_t buf[4];
    int ret;

    len = (uint32_t)data->length;
    _krb5_put_int(buf, len, 4);
    if (krb5_net_write (context, p_fd, buf, 4) != 4
	|| krb5_net_write (context, p_fd, data->data, len) != (ssize_t)len) {
	ret = errno;
	krb5_set_error_message (context, ret, "write: %s", strerror(ret));
	return ret;
    }
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_write_priv_message(krb5_context context,
			krb5_auth_context ac,
			krb5_pointer p_fd,
			krb5_data *data)
{
    krb5_error_code ret;
    krb5_data packet;

    ret = krb5_mk_priv (context, ac, data, &packet, NULL);
    if(ret)
	return ret;
    ret = krb5_write_message(context, p_fd, &packet);
    krb5_data_free(&packet);
    return ret;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_write_safe_message(krb5_context context,
			krb5_auth_context ac,
			krb5_pointer p_fd,
			krb5_data *data)
{
    krb5_error_code ret;
    krb5_data packet;
    ret = krb5_mk_safe (context, ac, data, &packet, NULL);
    if(ret)
	return ret;
    ret = krb5_write_message(context, p_fd, &packet);
    krb5_data_free(&packet);
    return ret;
}
