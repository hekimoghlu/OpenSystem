/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
krb5_build_ap_req (krb5_context context,
		   krb5_enctype enctype,
		   krb5_creds *cred,
		   krb5_flags ap_options,
		   krb5_data authenticator,
		   krb5_data *retdata)
{
  krb5_error_code ret = 0;
  AP_REQ ap;
  size_t len;

  ap.pvno = 5;
  ap.msg_type = krb_ap_req;

  memset(&ap.ap_options, 0, sizeof(ap.ap_options));
  ap.ap_options.use_session_key = (ap_options & AP_OPTS_USE_SESSION_KEY) > 0;
  ap.ap_options.mutual_required = (ap_options & AP_OPTS_MUTUAL_REQUIRED) > 0;

  ret = decode_Ticket(cred->ticket.data, cred->ticket.length, &ap.ticket, &len);
  if (ret) {
      krb5_data_zero(retdata);
      return ret;
  }

  ap.authenticator.etype = enctype;
  ap.authenticator.kvno  = NULL;
  ap.authenticator.cipher = authenticator;

  ASN1_MALLOC_ENCODE(AP_REQ, retdata->data, retdata->length,
		     &ap, &len, ret);
  if(ret == 0 && retdata->length != len)
      krb5_abortx(context, "internal error in ASN.1 encoder");
  else if (ret)
      krb5_data_zero(retdata);

  free_AP_REQ(&ap);
  return ret;
}
