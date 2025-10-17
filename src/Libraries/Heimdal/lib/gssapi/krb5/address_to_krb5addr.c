/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
#include "gsskrb5_locl.h"

#include <roken.h>

krb5_error_code
_gsskrb5i_address_to_krb5addr(krb5_context context,
			      OM_uint32 gss_addr_type,
			      gss_buffer_desc *gss_addr,
			      int16_t port,
			      krb5_address *address)
{
   int addr_type;
   struct sockaddr sa;
   krb5_socklen_t sa_size = sizeof(sa);
   krb5_error_code problem;

   if (gss_addr == NULL)
      return GSS_S_FAILURE;

   switch (gss_addr_type) {
#ifdef HAVE_IPV6
      case GSS_C_AF_INET6: addr_type = AF_INET6;
                           break;
#endif /* HAVE_IPV6 */

      case GSS_C_AF_INET:  addr_type = AF_INET;
                           break;
      default:
                           return GSS_S_FAILURE;
   }

   problem = krb5_h_addr2sockaddr (context,
				   addr_type,
                                   gss_addr->value,
                                   &sa,
                                   &sa_size,
                                   port);
   if (problem)
      return GSS_S_FAILURE;

   problem = krb5_sockaddr2address (context, &sa, address);

   return problem;
}
