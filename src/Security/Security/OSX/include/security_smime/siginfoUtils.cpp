/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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
 * siginfoUtils.cpp - private C++ routines for cmssiginfo
 */

#include <Security/SecCmsSignerInfo.h>
#include <security_utilities/simpleprefs.h>
#include "cmspriv.h" /* prototype */

/*
 * RFC 3278 section section 2.1.1 states that the signatureAlgorithm 
 * field contains the full ecdsa-with-SHA1 OID, not plain old ecPublicKey 
 * as would appear in other forms of signed datas. However Microsoft doesn't 
 * do this, it puts ecPublicKey there, and if we put ecdsa-with-SHA1 there, 
 * MS can't verify - presumably because it takes the digest of the digest 
 * before feeding it to ECDSA.
 * We handle this with a preference; default if it's not there is OFF
 */

bool SecCmsMsEcdsaCompatMode()
{
    bool msCompat = false;
    Dictionary* pd =
        Dictionary::CreateDictionary(kMSCompatibilityDomain, Dictionary::US_User, false);
    if(pd == NULL) {
        pd = Dictionary::CreateDictionary(kMSCompatibilityDomain, Dictionary::US_System, false);
    }
    if(pd != NULL) {
        msCompat = pd->getBoolValue(kMSCompatibilityMode);
        delete pd;
    }
    return msCompat;
}
