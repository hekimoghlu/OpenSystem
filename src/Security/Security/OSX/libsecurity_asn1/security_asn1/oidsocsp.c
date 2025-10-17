/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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
 File:      oidsocsp.cpp

 Contains:  Object Identifiers for OCSP
 */

#include "oidsocsp.h"
#include <Security/oidsbase.h>

SECASN1OID_DEF(OID_PKIX_OCSP, OID_AD_OCSP);
SECASN1OID_DEF(OID_PKIX_OCSP_BASIC, OID_AD_OCSP, 1);
SECASN1OID_DEF(OID_PKIX_OCSP_NONCE, OID_AD_OCSP, 2);
SECASN1OID_DEF(OID_PKIX_OCSP_CRL, OID_AD_OCSP, 3);
SECASN1OID_DEF(OID_PKIX_OCSP_RESPONSE, OID_AD_OCSP, 4);
SECASN1OID_DEF(OID_PKIX_OCSP_NOCHECK, OID_AD_OCSP, 5);
SECASN1OID_DEF(OID_PKIX_OCSP_ARCHIVE_CUTOFF, OID_AD_OCSP, 6);
SECASN1OID_DEF(OID_PKIX_OCSP_SERVICE_LOCATOR, OID_AD_OCSP, 7);

SECASN1OID_DEF(OID_GOOGLE_OCSP_SCT, GOOGLE_OCSP_SCT_OID);
