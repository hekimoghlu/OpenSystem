/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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

#ifndef HEIMDAL_SMALLER

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_decode_EncTicketPart (krb5_context context,
			   const void *data,
			   size_t length,
			   EncTicketPart *t,
			   size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return decode_EncTicketPart(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_encode_EncTicketPart (krb5_context context,
			   void *data,
			   size_t length,
			   EncTicketPart *t,
			   size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return encode_EncTicketPart(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_decode_EncASRepPart (krb5_context context,
			  const void *data,
			  size_t length,
			  EncASRepPart *t,
			  size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return decode_EncASRepPart(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_encode_EncASRepPart (krb5_context context,
			  void *data,
			  size_t length,
			  EncASRepPart *t,
			  size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return encode_EncASRepPart(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_decode_EncTGSRepPart (krb5_context context,
			   const void *data,
			   size_t length,
			   EncTGSRepPart *t,
			   size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return decode_EncTGSRepPart(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_encode_EncTGSRepPart (krb5_context context,
			   void *data,
			   size_t length,
			   EncTGSRepPart *t,
			   size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return encode_EncTGSRepPart(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_decode_EncAPRepPart (krb5_context context,
			  const void *data,
			  size_t length,
			  EncAPRepPart *t,
			  size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return decode_EncAPRepPart(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_encode_EncAPRepPart (krb5_context context,
			  void *data,
			  size_t length,
			  EncAPRepPart *t,
			  size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return encode_EncAPRepPart(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_decode_Authenticator (krb5_context context,
			   const void *data,
			   size_t length,
			   Authenticator *t,
			   size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return decode_Authenticator(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_encode_Authenticator (krb5_context context,
			   void *data,
			   size_t length,
			   Authenticator *t,
			   size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return encode_Authenticator(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_decode_EncKrbCredPart (krb5_context context,
			    const void *data,
			    size_t length,
			    EncKrbCredPart *t,
			    size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return decode_EncKrbCredPart(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_encode_EncKrbCredPart (krb5_context context,
			    void *data,
			    size_t length,
			    EncKrbCredPart *t,
			    size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return encode_EncKrbCredPart (data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_decode_ETYPE_INFO (krb5_context context,
			const void *data,
			size_t length,
			ETYPE_INFO *t,
			size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return decode_ETYPE_INFO(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_encode_ETYPE_INFO (krb5_context context,
			void *data,
			size_t length,
			ETYPE_INFO *t,
			size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return encode_ETYPE_INFO (data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_decode_ETYPE_INFO2 (krb5_context context,
			const void *data,
			size_t length,
			ETYPE_INFO2 *t,
			size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return decode_ETYPE_INFO2(data, length, t, len);
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_encode_ETYPE_INFO2 (krb5_context context,
			 void *data,
			 size_t length,
			 ETYPE_INFO2 *t,
			 size_t *len)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    return encode_ETYPE_INFO2 (data, length, t, len);
}

#endif /* HEIMDAL_SMALLER */
