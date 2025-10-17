/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

OM_uint32 GSSAPI_CALLCONV _gsskrb5_indicate_mechs
           (OM_uint32 * minor_status,
            gss_OID_set * mech_set
           )
{
  OM_uint32 ret, junk;

  ret = gss_create_empty_oid_set(minor_status, mech_set);
  if (ret)
      return ret;

  ret = gss_add_oid_set_member(minor_status, GSS_KRB5_MECHANISM, mech_set);
  if (ret) {
      gss_release_oid_set(&junk, mech_set);
      return ret;
  }

  *minor_status = 0;
  return GSS_S_COMPLETE;
}
