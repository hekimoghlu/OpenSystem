/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#ifndef GSSAPI_SCRAM_H_
#define GSSAPI_SCRAM_H_

#include <gssapi.h>

GSSAPI_CPP_START

extern gss_OID_desc GSSAPI_LIB_VARIABLE __gss_scram_mechanism_oid_desc;
#define GSS_SCRAM_MECHANISM (&__gss_scram_mechanism_oid_desc)

GSSAPI_CPP_END

#endif /* GSSAPI_SPNEGO_H_ */
