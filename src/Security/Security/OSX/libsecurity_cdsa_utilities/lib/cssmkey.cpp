/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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
//
// PODWrapper for CssmKey and related types
//
#include <security_cdsa_utilities/cssmkey.h>


//
// Methods for the CssmKey class
//
CssmKey::CssmKey(const CSSM_KEY &key)
{
	KeyHeader = key.KeyHeader;
    KeyData = key.KeyData;
}

CssmKey::CssmKey(const CSSM_DATA &keyData)
{
	clearPod();
    KeyData = keyData;
    KeyHeader.HeaderVersion = CSSM_KEYHEADER_VERSION;
    KeyHeader.BlobType = CSSM_KEYBLOB_RAW;
    KeyHeader.Format = CSSM_KEYBLOB_RAW_FORMAT_NONE;
}

CssmKey::CssmKey(uint32 length, void *data)
{
	clearPod();
	KeyData = CssmData(data, length);
    KeyHeader.HeaderVersion = CSSM_KEYHEADER_VERSION;
    KeyHeader.BlobType = CSSM_KEYBLOB_RAW;
    KeyHeader.Format = CSSM_KEYBLOB_RAW_FORMAT_NONE;
}
