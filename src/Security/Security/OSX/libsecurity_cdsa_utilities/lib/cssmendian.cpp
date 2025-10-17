/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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
// Byte order ("endian-ness") handling
//
#include <security_cdsa_utilities/cssmendian.h>

namespace Security {


void n2hi(CssmKey::Header &header)
{
		header.HeaderVersion = n2h(header.HeaderVersion);
		header.BlobType = n2h(header.BlobType);
		header.Format = n2h(header.Format);
		header.AlgorithmId = n2h(header.AlgorithmId);
		header.KeyClass = n2h(header.KeyClass);
		header.LogicalKeySizeInBits = n2h(header.LogicalKeySizeInBits);
		header.KeyAttr = n2h(header.KeyAttr);
		header.KeyUsage = n2h(header.KeyUsage);
		header.WrapAlgorithmId = n2h(header.WrapAlgorithmId);
		header.WrapMode = n2h(header.WrapMode);
		header.Reserved = n2h(header.Reserved);
}

void h2ni(CssmKey::Header &key)
{
		n2hi(key);
}

}	// end namespace Security
